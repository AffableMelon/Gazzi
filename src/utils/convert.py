from pdf2image import convert_from_path
import cv2
import re
from pathlib import Path
import numpy as np
from typing import Tuple, Optional, List


def preprocess_image_gray(pil_image) -> np.ndarray:
    """
    Upscale, Grayscale, Contrast, Deskew.
    Returns a CLAHE-enhanced grayscale image.
    """
    image = np.array(pil_image)

    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Upscale (Crucial for dense Amharic characters and numbers)
    # 1.5x scaling gives Tesseract more pixel clarity for fine strokes
    # h, w = gray.shape
    # gray = cv2.resize(gray, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)

    h, w = gray.shape
    gray = cv2.resize(gray, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_LANCZOS4)

    # 3. Denoise (Keep it very light so we don't blur character edges)
    # denoised = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=21)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75) 
    # 4. CLAHE (Boosts contrast without destroying thin lines like hard thresholding does)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # contrast = clahe.apply(denoised)

    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 31, 10
    )

    # Subtle sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    contrast = cv2.filter2D(binary, -1, kernel)

    # 5. Deskew
    deskewed = _deskew(contrast)

    # Note: We do NOT binarize here. We return the crisp grayscale image.
    return deskewed


def _deskew(image):
    """
    Detect skew angle via Hough line transform and rotate to correct.
    Only corrects small angles (< 10°) to avoid making things worse.
    """
    # Edge detection for line finding
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Hough lines — detect dominant text line angle
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=100,
                            minLineLength=image.shape[1] // 4,
                            maxLineGap=20)

    if lines is None or len(lines) == 0:
        return image

    # Compute median angle of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines (text lines)
            if abs(angle) < 10:
                angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)

    # Only correct if skew is noticeable but not extreme
    if abs(median_angle) < 0.3:
        return image  # essentially straight, skip rotation

    # Rotate around center
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _remove_noise_blobs(binary_image, min_area=30):
    """
    Remove connected components smaller than min_area pixels.
    These are speckle noise that Tesseract reads as stray characters.
    """
    # Invert: connectedComponents expects white objects on black
    inverted = cv2.bitwise_not(binary_image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    # Build mask of noise blobs
    noise_mask = np.zeros_like(binary_image)
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            noise_mask[labels == i] = 255

    # Set noise regions to white (background) in original
    cleaned = binary_image.copy()
    cleaned[noise_mask == 255] = 255
    return cleaned


def find_column_divider_line(image) -> Tuple[int, bool]:
    """
    Detect the vertical ruling line between columns using morphology.

    This finds actual printed lines, not text gaps.
    Works on both dense and sparse pages because the line
    is always present regardless of text content.
    """
    h, w = image.shape[:2]

    # --- Step 1: Isolate vertical line structures ---

    vertical_kernel_height = h // 4  # must be at least 25% of page height
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_kernel_height)  # 1px wide, very tall
    )

    # --- Step 1.5: Binarize specifically for line detection ---
    # The input 'image' is grayscale. Morphology works best on binary
    # for structure extraction to avoid noise counting as signal.
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Otsu's thresholding to get strict binary (white lines on black bg)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening: erode then dilate
    # This REMOVES everything that isn't tall and narrow
    # Text disappears. Only vertical lines survive.
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # --- Step 2: Search only the middle region ---
    # Widen search area to account for off-center scans
    search_left = int(w * 0.20)
    search_right = int(w * 0.80)
    search_region = vertical_lines[:, search_left:search_right]

    # --- Step 3: Project vertically (sum each column) ---
    # The divider line will have a strong peak
    projection = np.sum(search_region > 0, axis=0)

    # --- Step 4: Find the strongest peak ---
    if np.max(projection) < h * 0.10:
        # No vertical line found that spans at least 10% of page
        # Fall back to midpoint
        return w // 2, False  # (x_position, was_line_found)

    # The peak in the projection = the divider line's x position
    line_local_x = np.argmax(projection)
    line_x = search_left + line_local_x

    return line_x, True  # (x_position, was_line_found)


def split_image_columns(image: np.ndarray):
    h, w = image.shape[:2]

    line_x, found_line = find_column_divider_line(image)

    if found_line:
        # Include the line in both halves (overlap by a few pixels)
        overlap = 3  # pixels of overlap on each side of the line
        left_img = image[:, :min(w, line_x + overlap)]
        right_img = image[:, max(0, line_x - overlap):]
    else:
        # No line found — fall back to midpoint with small overlap
        mid = w // 2
        overlap = 5
        left_img = image[:, :min(w, mid + overlap)]
        right_img = image[:, max(0, mid - overlap):]

    return left_img, right_img

def find_header_separator_y(binary_image: np.ndarray,
                            min_line_length_ratio: float = 0.6,
                            line_thickness_max: int = 8) -> Optional[int]:
    """
    Returns y of the horizontal header separator line (best candidate),
    or None if not found.
    """
    h, w = binary_image.shape

    # Invert so lines/text become white (255) on black background for morphology
    inv = cv2.bitwise_not(binary_image)

    # Kernel width controls what counts as a "long horizontal line"
    kernel_w = max(30, int(w * min_line_length_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))

    # Extract horizontal lines: opening removes non-horizontal components
    lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)

    # Optional: thicken slightly to connect broken line segments
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=1)

    # Find connected components of line candidates
    num, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)

    candidates: List[int] = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        # Filter: must be long and thin-ish
        if bw >= int(w * min_line_length_ratio) and bh <= line_thickness_max:
            candidates.append(y + bh // 2)

    if not candidates:
        return None

    # Usually the header separator is the lowest detected line near the top half
    candidates = [yy for yy in candidates if yy < int(h * 0.6)]
    if not candidates:
        return None
    print(candidates)
    return max(candidates)  # choose the lowest one (closest to body)

def split_image_header_by_line(binary_or_gray: np.ndarray, y: int, pad: int = 10):
    h = binary_or_gray.shape[0]
    y0 = max(0, y - pad)
    header = binary_or_gray[:y0, :]
    body   = binary_or_gray[min(h, y + pad):, :]
    return header, body

def pdf_to_images(pdf_path: str, dpi=300):
    """Convert PDF to list of PIL images at specified DPI."""
    return convert_from_path(pdf_path, dpi=dpi)


def upscale_if_needed(image, min_height_per_line=40, estimated_lines=50):
    """
    If the image resolution is too low for good OCR, upscale it.
    Tesseract works best when text line height is ~40-60px.
    """
    h, w = image.shape[:2]
    estimated_line_height = h / estimated_lines

    if estimated_line_height < min_height_per_line:
        scale = min_height_per_line / estimated_line_height
        scale = min(scale, 3.0)  # cap at 3x to avoid huge images
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_CUBIC)
    return image


def main():
    input_dir = Path("./data/raw")
    output_dir = Path("./data/processed/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # file = "data/raw/አዋጅ ቁጥር 23-1988(Kibreab).pdf"
    file = "data/raw/አዋጅ ቁጥር 19-1988(Tsega).pdf"
    images = pdf_to_images(file)
    i = -1
    pil_img = np.array(images[0])
    # for i, pil_img in enumerate(images):
    processed_img = preprocess_image_gray(pil_img)
    p = preprocess_image(pil_img)
    filename = str(output_dir / f"አዋጅ ቁጥር 23-1988(Kibreab)_{i}")
    header_y = find_header_separator_y(p)
    header, body = split_image_header_by_line(p, header_y) 
    l, r = split_image_columns(body)
    cv2.imwrite(f"{filename}_left.png", l)
    cv2.imwrite(f"{filename}_right.png", r)
    cv2.imwrite(f"{filename}_header.png", header)
    cv2.imwrite(f"{filename}_.png", p)
    cv2.imwrite(f"{filename}_ng.png", processed_img)

    print(f"Saved {filename} ({processed_img.shape[1]}x{processed_img.shape[0]})")


    # for pdf_path in sorted(input_dir.iterdir()):
    #     if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
    #         continue

    #     match = re.search(r"አዋጅ.*\.pdf", pdf_path.name)
    #     if match:
    #         clean_base = re.sub(r"[\s\.\(\)]+", "_", match.group(0).replace(".pdf", ""))
    #     else:
    #         clean_base = pdf_path.stem

    #     try:
    #         images = pdf_to_images(str(pdf_path))
    #     except Exception as e:
    #         print(f"Failed to convert {pdf_path}: {e}")
    #         continue

    #     for i, pil_img in enumerate(images):
    #         processed_img = preprocess_image(pil_img)
    #         filename = str(output_dir / f"{clean_base}_{i}")
    #         l, r = split_image_columns(processed_img)
    #         cv2.imwrite(f"{filename}_left.png", l)
    #         cv2.imwrite(f"{filename}_right.png", r)
    #         print(f"Saved {filename} ({processed_img.shape[1]}x{processed_img.shape[0]})")


if __name__ == "__main__":
    main()

from pdf2image import convert_from_path
import cv2
import re
from pathlib import Path
import numpy as np


def preprocess_image(pil_image, for_ocr_lang="amh"):
    """
    Robust preprocessing pipeline:
    1. Convert to grayscale
    2. Denoise (preserve Amharic fine strokes)
    3. CLAHE for contrast normalization
    4. Deskew
    5. Light adaptive threshold (only for severely degraded scans)
    6. Morphological cleanup to remove speckle noise

    Returns a cleaned grayscale (not binary) NumPy image — Tesseract's
    LSTM engine (OEM 1) works better on grayscale than hard binary.
    """
    image = np.array(pil_image)

    # --- 1. Grayscale ---
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --- 2. Denoise (non-local means — preserves edges/strokes) ---
    # h=10 is moderate; for very noisy scans increase to 15-20
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                         searchWindowSize=21)

    # --- 3. CLAHE (contrast-limited adaptive histogram equalization) ---
    # Normalizes uneven lighting from scans without blowing out strokes
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    # --- 4. Deskew ---
    deskewed = _deskew(contrast)

    # --- 5. Conditional binarization ---
    # Only binarize if the image is severely degraded.
    # For most scanned PDFs at 300 DPI, Tesseract LSTM works better
    # on clean grayscale. We check contrast range as a heuristic.
    std = np.std(deskewed)
    if std < 40:
        # Very low contrast — likely a faded/stained scan; binarize gently
        binary = cv2.adaptiveThreshold(
            deskewed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,   # large block = gentler, preserves strokes
            C=15            # higher C = less aggressive black
        )
    else:
        # Good contrast — use Otsu on the already-enhanced image
        # This gives a cleaner result than adaptive for good scans
        _, binary = cv2.threshold(deskewed, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 6. Morphological noise removal ---
    # Remove small speckles that Tesseract reads as stray characters
    binary = _remove_noise_blobs(binary, min_area=30)

    return binary


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


def find_column_gutter(image):
    """
    Find the vertical gutter between two text columns using
    vertical projection profile (sum of white pixels per column).

    Returns the x-coordinate of the gutter center.
    Falls back to midpoint if no clear gutter is found.
    """
    h, w = image.shape[:2]

    # Only search the middle 30% of the image width for the gutter
    search_left = int(w * 0.35)
    search_right = int(w * 0.65)

    # Vertical projection: count white pixels in each column
    # (in a binary image, white=255 is background)
    if np.mean(image) > 127:
        # White background, dark text
        projection = np.sum(image[:, search_left:search_right] == 255, axis=0)
    else:
        projection = np.sum(image[:, search_left:search_right] == 0, axis=0)

    # Smooth the projection to avoid noise
    kernel_size = max(15, w // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(projection.astype(np.float32).reshape(1, -1),
                                 (kernel_size, 1), 0).flatten()

    # The gutter is where projection is maximum (most white/background pixels)
    gutter_local = np.argmax(smoothed)
    gutter_x = search_left + gutter_local

    # Validate: the gutter should have significantly more background than
    # the text regions. If not, fall back to midpoint.
    gutter_value = smoothed[gutter_local]
    mean_value = np.mean(smoothed)
    if gutter_value < mean_value * 1.1:
        # No clear gutter found — fall back
        return w // 2

    return gutter_x


def split_image_columns(image: np.ndarray):
    """
    Split image into two columns using detected gutter position.
    Adds a small margin around the gutter to avoid cutting into text.
    """
    h, w = image.shape[:2]
    gutter_x = find_column_gutter(image)

    # Add margin around gutter (don't cut into edge characters)
    margin = max(10, w // 100)
    left_img = image[:, :max(0, gutter_x - margin)]
    right_img = image[:, min(w, gutter_x + margin):]

    return left_img, right_img


def find_header_boundary(image):
    """
    Detect the header/body boundary using horizontal projection profile.
    The header typically ends at a horizontal line or a large gap in text.

    Returns the y-coordinate of the boundary.
    Falls back to a percentage-based estimate if no clear boundary is found.
    """
    h, w = image.shape[:2]

    # Only search the top 40% of the page for header boundary
    search_region = image[:int(h * 0.4), :]

    # Horizontal projection: count dark pixels per row
    if np.mean(image) > 127:
        row_dark = np.sum(search_region < 128, axis=1)
    else:
        row_dark = np.sum(search_region > 128, axis=1)

    # Smooth
    kernel = max(11, h // 80)
    if kernel % 2 == 0:
        kernel += 1
    smoothed = cv2.GaussianBlur(row_dark.astype(np.float32).reshape(-1, 1),
                                 (1, kernel), 0).flatten()

    # Look for a significant gap (low dark pixel count = empty row)
    threshold = np.mean(smoothed) * 0.15

    # Scan from top: find the first large gap after some content
    found_content = False
    gap_start = None
    min_gap_height = max(20, h // 50)

    for y in range(len(smoothed)):
        if smoothed[y] > threshold:
            found_content = True
            gap_start = None
        elif found_content:
            if gap_start is None:
                gap_start = y
            elif y - gap_start > min_gap_height:
                return y  # Found a significant gap — this is the header boundary

    # Fallback: use ~15% of page height (typical for gazette headers)
    return int(h * 0.15)


def split_image_header(image: np.ndarray, split_y: int = None):
    """
    Split image into header and body.
    If split_y is None, auto-detect the boundary.
    """
    h, w = image.shape[:2]

    if split_y is None:
        split_y = find_header_boundary(image)

    if h <= split_y or split_y < 50:
        return image, None

    header = image[:split_y, :]
    body = image[split_y:, :]
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
    pdf_path = Path("./data/raw/አዋጅ ቁጥር 23-1988(Kibreab).pdf")
    output_dir = Path("./data/processed/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    match = re.search(r"አዋጅ.*\.pdf", pdf_path.name)
    if match:
        clean_base = re.sub(r"[\s\.\(\)]+", "_", match.group(0).replace(".pdf", ""))
    else:
        clean_base = pdf_path.stem

    images = pdf_to_images(pdf_path)

    for i, pil_img in enumerate(images):
        processed_img = preprocess_image(pil_img)
        filename = str(output_dir / f"{clean_base}_{i}.png")
        cv2.imwrite(filename, processed_img)
        print(f"Saved {filename} ({processed_img.shape[1]}x{processed_img.shape[0]})")


if __name__ == "__main__":
    main()
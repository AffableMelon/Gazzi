from enum import Enum
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path

class Document:
    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.file_type = self._detect_type()

    def _detect_type(self):
        if self.path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            return "image"
        elif self.path.suffix.lower() == ".pdf":
            return "pdf"
        raise ValueError(f"Unsupported File Type, {self.path.suffix.lower()}")

    def is_pdf(self):
        return self._detect_type() == "pdf"

    def is_img(self):
        return self._detect_type() == "image"

class ProcessingStrategy(Enum):
    STANDARD = "standard"
    GRAYSCALE = "grayscale"

class DocProcessor:
    def __init__(self, strategy: str = "standard"):
        """
        Initialize the DocProcessor with a specific processing strategy.
        
        Args:
            strategy (str): The processing strategy to use. 
                            Options: 'standard', 'grayscale'.
                            Defaults to 'standard'.
        """
        self.strategy = strategy

    def process(self, document):
        """
        Process a document (PDF or image) into a stream of preprocessed images.
        """
        if document.is_pdf():
            for page in self.pdf_to_images(str(document.path)):
                yield self._process_page(page)
        else:
            images = Image.open(document.path)
            yield self._process_page(images)

    
    def pdf_to_images(self, pdf_path: str, dpi=300):
        """Convert PDF to list of PIL images at specified DPI."""
        return convert_from_path(pdf_path, dpi=dpi)

    def _process_page(self, pil_image):
        """
        Apply the selected preprocessing strategy to a single page image.
        """
        if self.strategy == ProcessingStrategy.GRAYSCALE.value:
            return self.preprocess_image_gray(pil_image)
        else:
            return self.preprocess_image(pil_image)

    def preprocess_image(self, pil_image):
        """
        Robust preprocessing pipeline:
        1. Convert to grayscale
        2. Denoise (preserve Amharic fine strokes)
        3. CLAHE for contrast normalization
        4. Deskew
        5. Light adaptive threshold (only for severely degraded scans)
        6. Morphological cleanup to remove speckle noise

        Returns a cleaned grayscale (not binary) NumPy image or binary if thresholded.
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
        deskewed = self._deskew(contrast)

        # --- 5. Conditional binarization ---
        # Only binarize if the image is severely degraded.
        # Check contrast range as a heuristic.
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
            _, binary = cv2.threshold(deskewed, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- 6. Morphological noise removal ---
        # Remove small speckles that Tesseract reads as stray characters
        binary = self._remove_noise_blobs(binary, min_area=30)

        return binary    

    def preprocess_image_gray(self, pil_image) -> np.ndarray:
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
        # 2x scaling gives Tesseract more pixel clarity for fine strokes
        h, w = gray.shape
        gray = cv2.resize(gray, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_LANCZOS4)

        # 3. Denoise (Keep it very light so we don't blur character edges)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75) 
        
        # 4. Adaptive Thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 31, 10
        )

        # Subtle sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        contrast = cv2.filter2D(binary, -1, kernel)

        # 5. Deskew
        deskewed = self._deskew(contrast)

        return deskewed
    
    def _deskew(self, image):
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
    
    def _remove_noise_blobs(self, binary_image, min_area=30):
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




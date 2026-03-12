import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

logger = logging.getLogger("legal_hil.pdf_extractor")

class PDFExtractor:
    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def get_document_info(self, pdf_path: str) -> Dict[str, Any]:
        """Returns page count and metadata."""
        doc = fitz.open(pdf_path)
        return {
            "page_count": len(doc),
            "title": doc.metadata.get("title", Path(pdf_path).stem),
            "file_size": Path(pdf_path).stat().st_size,
        }

    def render_page_to_png(self, pdf_path: str, page_number: int, dpi: Optional[int] = None) -> bytes:
        """
        Renders a 1-based page number to PNG bytes for LLM API or web display.
        """
        doc = fitz.open(pdf_path)
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"Page {page_number} out of range (1-{len(doc)})")

        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(dpi=dpi or self.dpi)
        return pix.tobytes("png")

    def render_page_to_numpy(self, pdf_path: str, page_number: int) -> np.ndarray:
        """Renders page to numpy array (H, W, 3)."""
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(dpi=self.dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return img

    def extract_text_and_spans(self, pdf_path: str, page_number: int) -> Dict[str, Any]:
        """
        Extracts native digital text spans and layout from PDF if searchable.
        """
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)
        text_dict = page.get_text("dict")
        plain_text = page.get_text("text")

        spans = []
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        if txt:
                            spans.append({
                                "text": txt,
                                "bbox": span["bbox"],
                                "font": span.get("font"),
                                "size": span.get("size")
                            })

        return {
            "page_number": page_number,
            "width": page.rect.width,
            "height": page.rect.height,
            "plain_text": plain_text,
            "spans": spans
        }

    def detect_layout_boundaries(self, pdf_path: str, page_number: int) -> Dict[str, Any]:
        """
        Detects vertical column divider and horizontal header divider
        using morphology (adapted from w2 layout splitter).
        """
        try:
            import cv2
            img = self.render_page_to_numpy(pdf_path, page_number)
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Detect vertical column divider line
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 5))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            search_left = int(w * 0.25)
            search_right = int(w * 0.75)
            search_region = vertical_lines[:, search_left:search_right]
            projection = np.sum(search_region > 0, axis=0)

            divider_found = False
            divider_x = w // 2
            if len(projection) > 0 and np.max(projection) > h * 0.10:
                divider_x = search_left + int(np.argmax(projection))
                divider_found = True

            # Detect horizontal header line
            inv = cv2.bitwise_not(binary)
            kernel_w = max(30, int(w * 0.40))
            horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
            horiz_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel)
            num, _, stats, _ = cv2.connectedComponentsWithStats(horiz_lines, connectivity=8)

            header_y = int(h * 0.12)
            header_found = False
            candidates = []
            for i in range(1, num):
                x, y, bw, bh, _ = stats[i]
                if bw >= int(w * 0.40) and bh <= 10 and y < int(h * 0.45):
                    candidates.append(y + bh // 2)

            if candidates:
                header_y = max(candidates)
                header_found = True

            return {
                "page_width": w,
                "page_height": h,
                "divider_x": divider_x,
                "divider_found": divider_found,
                "header_y": header_y,
                "header_found": header_found
            }
        except Exception as e:
            logger.warning(f"Layout detection error: {e}")
            return {
                "page_width": 1000,
                "page_height": 1400,
                "divider_x": 500,
                "divider_found": False,
                "header_y": 150,
                "header_found": False
            }

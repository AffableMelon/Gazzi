import json
import re
from pathlib import Path

from src.core.doc_processor import DocProcessor
from src.core.document import Document
from src.core.ocr_engine import Tesseract_OCREngine
from src.core.post_processor import PostProcessor
from src.utils.convert import (
    split_image_header,
    split_image_columns,
    upscale_if_needed,
)


class OCRPipeline:
    """
    End-to-end OCR pipeline for bilingual (Amharic/English) gazette documents.

    Flow:
    1. Load document → convert to page images
    2. Preprocess each page (deskew, denoise, CLAHE, threshold, noise removal)
    3. Auto-detect and split header from body
    4. Auto-detect column gutter and split into Amharic (left) / English (right)
    5. OCR each region with language-specific Tesseract config
    6. Post-process with language-aware cleaning
    7. Structure into JSON output
    """

    def __init__(self):
        self.processor = DocProcessor()
        self.ocr_engine = Tesseract_OCREngine()
        self.postprocessor = PostProcessor()

    def run(self, file_path: str):
        """
        Run the full pipeline on a document.

        :param file_path: path to PDF or image file
        :returns: dict with structured OCR results
        """
        document = Document(file_path)
        images = self.processor.process(document)

        pages = []
        full_text = ""

        for i, img in enumerate(images):
            page_result = self._process_page(img, page_num=i)
            pages.append(page_result)

            # Build running full text
            page_text = f"--- Page {i + 1} ---\n"
            if page_result.get("header"):
                page_text += f"[Header]\n{page_result['header']}\n\n"
            if page_result.get("amharic"):
                page_text += f"[Amharic]\n{page_result['amharic']}\n\n"
            if page_result.get("english"):
                page_text += f"[English]\n{page_result['english']}\n\n"

            full_text += page_text

        return {
            "pages": pages,
            "full_text": full_text.strip(),
            "page_count": len(pages),
            **self.postprocessor.structure(full_text),
        }

    def _process_page(self, img, page_num: int) -> dict:
        """
        Process a single page image.

        :param img: preprocessed numpy array (from DocProcessor)
        :param page_num: 0-indexed page number
        :returns: dict with header, amharic, english text
        """
        result = {
            "page": page_num + 1,
            "header": "",
            "amharic": "",
            "english": "",
        }

        # --- Header detection (first page, or any page with a header) ---
        header_text = ""
        if page_num == 0:
            # Auto-detect header boundary (no hardcoded y value)
            header_img, body_img = split_image_header(img, split_y=None)

            if header_img is not None and header_img.shape[0] > 50:
                header_upscaled = upscale_if_needed(header_img)
                raw_header = self.ocr_engine.extract_text(
                    header_upscaled, lang='eng+amh', psm=3
                )
                header_text = self.postprocessor.clean(raw_header)

            img_to_split = body_img if body_img is not None else img
        else:
            img_to_split = img

        # --- Column splitting (auto-detect gutter) ---
        left_img, right_img = split_image_columns(img_to_split)

        # --- OCR: Amharic (left column) ---
        if left_img is not None and left_img.size > 0:
            left_upscaled = upscale_if_needed(left_img)
            raw_amh = self.ocr_engine.extract_text(
                left_upscaled, lang='amh', psm=6
            )
            cleaned_amh = self.postprocessor.clean(raw_amh, lang='amh')
            result["amharic"] = cleaned_amh

        # --- OCR: English (right column) ---
        if right_img is not None and right_img.size > 0:
            right_upscaled = upscale_if_needed(right_img)
            raw_eng = self.ocr_engine.extract_text(
                right_upscaled, lang='eng', psm=6
            )
            cleaned_eng = self.postprocessor.clean(raw_eng, lang='eng')
            result["english"] = cleaned_eng

        result["header"] = header_text
        return result


# --- CLI entry point ---
if __name__ == "__main__":
    import sys
    pipeline = OCRPipeline()
    input_dir = Path("./data/raw")
    output_dir = Path("./data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(input_dir.iterdir()):
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            continue

        match = re.search(r"አዋጅ.*\.pdf", pdf_path.name)
        if match:
            clean_base = re.sub(r"[\s\.\(\)]+", "_", match.group(0).replace(".pdf", ""))
        else:
            clean_base = pdf_path.stem

        try:
            result = pipeline.run(pdf_path)
            with open(f"{output_dir}/{clean_base}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Failed to convert {pdf_path}: {e}")
            continue
        
        print(f"Output written to {output_dir}/{clean_base}.json")
        print(f"Total lines: {result['line_count']}")
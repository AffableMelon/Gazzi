import json
import re
from pathlib import Path

from src.core.doc_processor import DocProcessor, Document
from src.core.ocr_engine import Tesseract_OCREngine
from src.core.post_processor import PostProcessor
from src.core.layout_splitter import LayoutSplitter


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
        self.layout_splitter = LayoutSplitter()

    def run(self, file_path: str, full_text: bool = False) -> dict:
        """
        Run the full pipeline on a document.

        :param file_path: path to PDF or image file
        :returns: dict with structured OCR results
        """
        document = Document(file_path)
        image_gen = self.processor.process(document)

        pages = []
        # full_text = ""

        try:
            # Get the first image
            current_img = next(image_gen)
        except StopIteration:
            return {"pages": [], "full_text": ""} # Empty doc
        
        i = 0
        while True:
            try:
                # Try to grab the next image to see if current_img is the last one
                next_img = next(image_gen)
                
                # If this succeeds, current_img is NOT the last page
                page_result = self._process_page(current_img, page_num=i, is_last=False)
                current_img = next_img # Move pointer forward
                
            except StopIteration:
                # If next() fails, current_img is officially the LAST page
                page_result = self._process_page(current_img, page_num=i, is_last=True)
                pages.append(page_result)
                # full_text += self._build_page_text(page_result, i + 1)
                break # Exit loop
            
            pages.append(page_result)
            i += 1
            # full_text += self._build_page_text(page_result, i + 1)

        if full_text:
            full_text = self._build_full_text(pages)
            return {
            "pages": pages,
            "page_count": len(pages),
            "full_text": full_text.strip(),
            **self.postprocessor.structure(full_text),
        }

        return {
            "pages": pages,
            "page_count": len(pages),
        }
    
    def _build_full_text(self, pages):
        """
        Combine page-level text into a single full document text string.
        """
        full_text = ""
        for page in pages:
            full_text += f"--- PAGE {page['page']} ---\n"
            if page['header']:
                full_text += f"{page['header']}\n"
            if page['amharic']:
                full_text += f"{page['amharic']}\n"
            if page['english']:
                full_text += f"{page['english']}\n"
            full_text += "\n" # Add spacing between pages
        return full_text

    def _process_page(self, img, page_num: int, is_last=False) -> dict:
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
        amh_text = ""
        eng_text = ""
        page_layout = self.layout_splitter.split(img)
        
        if page_layout.has_header:
            raw = self.ocr_engine.extract_text(page_layout.header, lang="eng+amh", psm=6)
            header_text = self.postprocessor.clean(raw)

            
        if page_layout.has_columns:
             # --- Left column: Amharic ---
            if page_layout.left_column is not None:
                raw = self.ocr_engine.extract_text(page_layout.left_column, lang='amh', psm=4)
                amh_text = self.postprocessor.clean(raw, lang='amh')
                if is_last:
                    amh_text = self.postprocessor.truncate_tail_noise(amh_text, lang='amh')

            # --- Right column: English ---
            if page_layout.right_column is not None and page_layout.right_column.size > 0:
                raw = self.ocr_engine.extract_text(page_layout.right_column, lang='eng', psm=4)
                eng_text = self.postprocessor.clean(raw, lang='eng')
                if is_last:
                    eng_text = self.postprocessor.truncate_tail_noise(eng_text, lang='eng')

        result.update({"header": header_text, "amharic": amh_text, "english": eng_text})
        return result


# --- CLI entry point ---
if __name__ == "__main__":
    import sys
    pipeline = OCRPipeline()
    input_dir = Path("./data/raw")
    output_dir = Path("./data/output/ou-o")
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
        print(f"Total pages processed: {result['page_count']}")
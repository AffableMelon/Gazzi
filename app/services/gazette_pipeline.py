import io
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.models import (
    AlignedPair, HILSession, InputType, HILStatus, BlockType, AuditSummary
)
from app.services.pdf_extractor import PDFExtractor
from app.services.llm_service import LLMService
from app.services.audit_service import AuditService
from app.services.storage_service import StorageService

logger = logging.getLogger("legal_hil.gazette_pipeline")

class GazettePipeline:
    """
    Type 1 Pipeline: Ingests a single bilingual Ethiopian Negarit Gazette / Proclamation PDF
    with dual-column parallel Amharic & English text.
    """
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.extractor = PDFExtractor(dpi=150)
        self.llm = llm_service or LLMService()
        self.storage = StorageService()

    def process(
        self,
        pdf_path: str,
        page_start: int = 1,
        page_end: int = 3,
        document_id: Optional[str] = None,
        use_precomputed_if_available: bool = True
    ) -> HILSession:
        path = Path(pdf_path).resolve()
        doc_id = document_id or path.stem
        info = self.extractor.get_document_info(str(path))
        total_pages = info["page_count"]

        page_start = max(1, min(page_start, total_pages))
        page_end = max(page_start, min(page_end, total_pages))
        pages_to_process = list(range(page_start, page_end + 1))

        pairs: List[AlignedPair] = []
        metadata: Dict[str, Any] = {
            "source_pdf": path.name,
            "pdf_path": str(path),
            "pages_range": f"{page_start}-{page_end}",
            "extractor_mode": "LLM API" if self.llm.is_configured() else "Pre-indexed Gazette Extraction",
            "provider": self.llm.provider if self.llm.is_configured() else "offline_reference"
        }

        # Load precomputed extraction if available for this document
        if use_precomputed_if_available and not self.llm.is_configured():
            for example in self.storage.get_example_documents():
                if Path(example.pdf_path).name == path.name or str(Path(example.pdf_path).resolve()) == str(path.resolve()):
                    precomputed = self.storage.load_precomputed_data(example.id)
                    if precomputed:
                        logger.info(f"Using extraction data for {example.id}")
                        filtered = [p for p in precomputed if page_start <= p.page_number <= page_end]
                        pairs = filtered or precomputed
                        break

        # Fallback to local layout extraction if no precomputed found
        if not pairs:
            pairs = self._extract_pages(str(path), pages_to_process)

        # Audit and detect any discrepancies/hallucinations
        audit = AuditService.audit_pairs(pairs)

        session_id = str(uuid.uuid4())
        session = HILSession(
            session_id=session_id,
            document_id=doc_id,
            input_type=InputType.GAZETTE_BILINGUAL,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            file_name=path.name,
            total_pages=total_pages,
            pages_processed=pages_to_process,
            pairs=pairs,
            audit=audit,
            metadata=metadata
        )

        self.storage.save_session(session)
        return session

    def _extract_pages(self, pdf_path: str, pages: List[int]) -> List[AlignedPair]:
        pairs: List[AlignedPair] = []
        pair_id_counter = 1

        for page_num in pages:
            logger.info(f"Processing Gazette page {page_num} of {pdf_path}")

            if self.llm.is_configured():
                try:
                    img_bytes = self.extractor.render_page_to_png(pdf_path, page_num, dpi=180)
                    response = self.llm.extract_from_page_image(img_bytes)
                    if "blocks" in response and isinstance(response["blocks"], list):
                        for block in response["blocks"]:
                            b_type = self._map_type(block.get("type", "paragraph"))
                            pairs.append(AlignedPair(
                                id=f"p{page_num}_{pair_id_counter}",
                                line_id=block.get("line_id", pair_id_counter),
                                article_number=str(block.get("article_number", "")) or None,
                                type=b_type,
                                page_number=page_num,
                                amharic=block.get("amharic", "").strip(),
                                english=block.get("english", "").strip(),
                                confidence=float(block.get("confidence", 0.95)),
                                status=HILStatus.PENDING
                            ))
                            pair_id_counter += 1
                        continue
                except Exception as e:
                    logger.error(f"LLM page extraction failed for page {page_num}: {e}")

            page_pairs = self._local_layout_extract(pdf_path, page_num, pair_id_counter)
            pairs.extend(page_pairs)
            pair_id_counter += len(page_pairs)

        return pairs

    def _local_layout_extract(self, pdf_path: str, page_num: int, start_counter: int) -> List[AlignedPair]:
        page_pairs: List[AlignedPair] = []
        spans_info = self.extractor.extract_text_and_spans(pdf_path, page_num)
        layout = self.extractor.detect_layout_boundaries(pdf_path, page_num)

        w = spans_info["width"]
        divider_x = layout["divider_x"] * (w / layout["page_width"]) if layout["page_width"] else w / 2
        header_y = layout["header_y"] * (spans_info["height"] / layout["page_height"]) if layout["page_height"] else 120

        header_lines: List[str] = []
        left_lines: List[str] = []
        right_lines: List[str] = []

        for span in spans_info["spans"]:
            bbox = span["bbox"]
            y0 = bbox[1]
            x0 = bbox[0]
            txt = span["text"].strip()
            if not txt:
                continue

            if y0 < header_y:
                header_lines.append(txt)
            elif x0 < divider_x:
                left_lines.append(txt)
            else:
                right_lines.append(txt)

        # Scanned page fallback: If no digital spans were extracted, run Tesseract OCR
        if not spans_info["spans"]:
            try:
                import pytesseract
                from PIL import Image
                img_bytes = self.extractor.render_page_to_png(pdf_path, page_num, dpi=180)
                page_img = Image.open(io.BytesIO(img_bytes))
                iw, ih = page_img.size
                div_pixel = int(layout["divider_x"] * (iw / layout["page_width"])) if layout["page_width"] else iw // 2
                hdr_pixel = int(layout["header_y"] * (ih / layout["page_height"])) if layout["page_height"] else int(ih * 0.12)

                left_crop = page_img.crop((0, hdr_pixel, div_pixel, ih))
                right_crop = page_img.crop((div_pixel, hdr_pixel, iw, ih))

                left_text = pytesseract.image_to_string(left_crop, lang="amh")
                right_text = pytesseract.image_to_string(right_crop, lang="eng")

                left_lines = [l.strip() for l in left_text.splitlines() if l.strip()]
                right_lines = [l.strip() for l in right_text.splitlines() if l.strip()]
            except Exception as e:
                logger.warning(f"Local OCR fallback error on page {page_num}: {e}")

        counter = start_counter

        if header_lines:
            page_pairs.append(AlignedPair(
                id=f"p{page_num}_{counter}",
                line_id=counter,
                type=BlockType.HEADER,
                page_number=page_num,
                amharic=" ".join(header_lines[:len(header_lines)//2 or 1]),
                english=" ".join(header_lines[len(header_lines)//2 or 1:]),
                confidence=0.92,
                status=HILStatus.PENDING
            ))
            counter += 1

        amh_blocks = self._chunk_lines_to_paragraphs(left_lines)
        eng_blocks = self._chunk_lines_to_paragraphs(right_lines)

        max_blocks = max(len(amh_blocks), len(eng_blocks))
        for i in range(max_blocks):
            amh = amh_blocks[i] if i < len(amh_blocks) else ""
            eng = eng_blocks[i] if i < len(eng_blocks) else ""

            is_article = "Article" in eng or "አንቀጽ" in amh or "Art." in eng

            page_pairs.append(AlignedPair(
                id=f"p{page_num}_{counter}",
                line_id=counter,
                type=BlockType.ARTICLE if is_article else BlockType.PARAGRAPH,
                page_number=page_num,
                amharic=amh,
                english=eng,
                confidence=0.90 if (amh and eng) else 0.50,
                status=HILStatus.PENDING if (amh and eng) else HILStatus.FLAGGED
            ))
            counter += 1

        return page_pairs

    def _chunk_lines_to_paragraphs(self, lines: List[str]) -> List[str]:
        blocks = []
        cur = []
        for line in lines:
            if line.endswith(".") or line.endswith("።") or len(line) < 25:
                cur.append(line)
                blocks.append(" ".join(cur))
                cur = []
            else:
                cur.append(line)
        if cur:
            blocks.append(" ".join(cur))
        return blocks or [" ".join(lines)] if lines else []

    def _map_type(self, raw: str) -> BlockType:
        raw = raw.lower()
        if "head" in raw: return BlockType.HEADER
        if "tit" in raw: return BlockType.TITLE
        if "art" in raw: return BlockType.ARTICLE
        if "sub" in raw: return BlockType.SUB_ARTICLE
        if "toc" in raw: return BlockType.TOC
        return BlockType.PARAGRAPH

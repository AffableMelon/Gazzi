import io
import re
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.models import (
    AlignedPair, HILSession, InputType, HILStatus, BlockType
)
from app.services.pdf_extractor import PDFExtractor
from app.services.llm_service import LLMService
from app.services.audit_service import AuditService
from app.services.storage_service import StorageService

logger = logging.getLogger("legal_hil.dual_code_pipeline")

class DualCodePipeline:
    """
    Type 2 Pipeline: Ingests two separate PDFs — one in Amharic and one in English —
    such as civil codes, criminal codes, or procedural codes, and aligns them
    article-by-article and clause-by-clause into a parallel corpus.
    """
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.extractor = PDFExtractor(dpi=150)
        self.llm = llm_service or LLMService()
        self.storage = StorageService()

    def process(
        self,
        pdf_path_amh: str,
        pdf_path_eng: str,
        page_start: int = 1,
        page_end: int = 5,
        document_id: Optional[str] = None,
        use_precomputed_if_available: bool = True
    ) -> HILSession:
        path_amh = Path(pdf_path_amh).resolve()
        path_eng = Path(pdf_path_eng).resolve()
        doc_id = document_id or f"{path_amh.stem}_aligned"

        info_amh = self.extractor.get_document_info(str(path_amh))
        info_eng = self.extractor.get_document_info(str(path_eng))
        total_pages = max(info_amh["page_count"], info_eng["page_count"])

        metadata: Dict[str, Any] = {
            "source_pdf": path_amh.name,
            "source_pdf_amharic": path_amh.name,
            "source_pdf_english": path_eng.name,
            "pdf_path": str(path_amh),
            "pdf_path_amh": str(path_amh),
            "pdf_path_eng": str(path_eng),
            "extractor_mode": "LLM API" if self.llm.is_configured() else "Pre-indexed Codebook Extraction",
            "provider": self.llm.provider if self.llm.is_configured() else "offline_reference"
        }

        pairs: List[AlignedPair] = []

        # Check precomputed extraction data if offline
        if use_precomputed_if_available and not self.llm.is_configured():
            for example in self.storage.get_example_documents():
                if example.input_type == InputType.CODEBOOK_DUAL:
                    if Path(example.pdf_path).name == path_amh.name or (example.pdf_path_eng and Path(example.pdf_path_eng).name == path_eng.name):
                        precomputed = self.storage.load_precomputed_data(example.id)
                        if precomputed:
                            logger.info(f"Using precomputed dual-code corpus for {example.id}")
                            pairs = precomputed
                            break

        if not pairs:
            pairs = self._extract_and_align_codebooks(str(path_amh), str(path_eng), page_start, page_end)

        audit = AuditService.audit_pairs(pairs)

        session_id = str(uuid.uuid4())
        session = HILSession(
            session_id=session_id,
            document_id=doc_id,
            input_type=InputType.CODEBOOK_DUAL,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            file_name=f"{path_amh.name} + {path_eng.name}",
            total_pages=total_pages,
            pages_processed=list(range(page_start, page_end + 1)),
            pairs=pairs,
            audit=audit,
            metadata=metadata
        )

        self.storage.save_session(session)
        return session

    def _extract_and_align_codebooks(
        self,
        pdf_path_amh: str,
        pdf_path_eng: str,
        page_start: int,
        page_end: int
    ) -> List[AlignedPair]:
        amh_texts = []
        for p in range(page_start, page_end + 1):
            try:
                info = self.extractor.extract_text_and_spans(pdf_path_amh, p)
                txt = info["plain_text"].strip()
                if not txt:
                    # Scanned page OCR fallback
                    import pytesseract
                    from PIL import Image
                    img_bytes = self.extractor.render_page_to_png(pdf_path_amh, p, dpi=180)
                    page_img = Image.open(io.BytesIO(img_bytes))
                    txt = pytesseract.image_to_string(page_img, lang="amh").strip()

                if txt:
                    amh_texts.append(txt)
            except Exception as e:
                logger.warning(f"Error extracting Amharic page {p}: {e}")

        eng_texts = []
        for p in range(page_start, page_end + 1):
            try:
                info = self.extractor.extract_text_and_spans(pdf_path_eng, p)
                txt = info["plain_text"].strip()
                if not txt:
                    # Scanned page OCR fallback
                    import pytesseract
                    from PIL import Image
                    img_bytes = self.extractor.render_page_to_png(pdf_path_eng, p, dpi=180)
                    page_img = Image.open(io.BytesIO(img_bytes))
                    txt = pytesseract.image_to_string(page_img, lang="eng").strip()

                if txt:
                    eng_texts.append(txt)
            except Exception as e:
                logger.warning(f"Error extracting English page {p}: {e}")

        full_amh = "\n\n".join(amh_texts)
        full_eng = "\n\n".join(eng_texts)

        if self.llm.is_configured() and full_amh and full_eng:
            try:
                resp = self.llm.align_texts(full_amh[:4000], full_eng[:4000])
                if "aligned_pairs" in resp and isinstance(resp["aligned_pairs"], list):
                    pairs = []
                    for idx, p in enumerate(resp["aligned_pairs"], start=1):
                        pairs.append(AlignedPair(
                            id=f"dual_{idx}",
                            line_id=p.get("line_id", idx),
                            article_number=str(p.get("article_number", "")) or None,
                            type=BlockType.ARTICLE if "art" in p.get("type", "").lower() else BlockType.PARAGRAPH,
                            amharic=p.get("amharic", ""),
                            english=p.get("english", ""),
                            confidence=float(p.get("confidence", 0.95)),
                            status=HILStatus.PENDING
                        ))
                    return pairs
            except Exception as e:
                logger.error(f"LLM alignment failed: {e}")

        return self._heuristic_article_align(full_amh, full_eng)

    def _heuristic_article_align(self, amh_text: str, eng_text: str) -> List[AlignedPair]:
        eng_articles = re.split(r"(?=Article\s+\d+|Art\.\s*\d+)", eng_text)
        amh_articles = re.split(r"(?=አንቀጽ\s+[፩-፱፲፳፴፵፶፷፸፹፺፻\d]+|ቍ\s*፡\s*[፩-፱፲፳፴፵፶፷፸፹፺፻\d]+)", amh_text)

        pairs: List[AlignedPair] = []
        max_len = max(len(amh_articles), len(eng_articles))

        line_id = 1
        for i in range(max_len):
            amh = amh_articles[i].strip() if i < len(amh_articles) else ""
            eng = eng_articles[i].strip() if i < len(eng_articles) else ""

            if not amh and not eng:
                continue

            art_match = re.search(r"(?:Article|Art\.)\s*(\d+)", eng, re.IGNORECASE)
            art_num = art_match.group(1) if art_match else None

            amh_lines = [l.strip() for l in amh.split("\n") if l.strip()]
            eng_lines = [l.strip() for l in eng.split("\n") if l.strip()]

            if len(amh_lines) == len(eng_lines) and len(amh_lines) > 1:
                for a_sub, e_sub in zip(amh_lines, eng_lines):
                    pairs.append(AlignedPair(
                        id=f"pair_{line_id}",
                        line_id=line_id,
                        article_number=art_num,
                        type=BlockType.ARTICLE if ("Art" in e_sub or "አንቀጽ" in a_sub) else BlockType.PARAGRAPH,
                        page_number=1 + (line_id // 10),
                        amharic=a_sub,
                        english=e_sub,
                        confidence=0.95,
                        status=HILStatus.PENDING
                    ))
                    line_id += 1
            else:
                conf = 0.90 if (amh and eng) else 0.45
                status = HILStatus.PENDING if (amh and eng) else HILStatus.FLAGGED
                reason = None
                if abs(len(amh_lines) - len(eng_lines)) > 2:
                    reason = f"Line count discrepancy: Amharic ({len(amh_lines)}) vs English ({len(eng_lines)})"
                    status = HILStatus.FLAGGED
                    conf = 0.65

                pairs.append(AlignedPair(
                    id=f"pair_{line_id}",
                    line_id=line_id,
                    article_number=art_num,
                    type=BlockType.ARTICLE if ("Article" in eng or "አንቀጽ" in amh) else BlockType.PARAGRAPH,
                    page_number=1 + (line_id // 10),
                    amharic=amh,
                    english=eng,
                    confidence=conf,
                    status=status,
                    discrepancy_reason=reason
                ))
                line_id += 1

        return pairs

import csv
import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz
from app.config import settings
from app.models import ExampleDocument, InputType, HILSession, AlignedPair, AuditSummary, HILStatus, BlockType

logger = logging.getLogger("legal_hil.storage")

class StorageService:
    def __init__(self):
        self.sessions_dir = settings.SESSIONS_DIR
        self.exports_dir = settings.EXPORTS_DIR
        self.uploads_dir = settings.UPLOADS_DIR
        self.precomputed_dir = settings.PRECOMPUTED_DIR
        self.base_dir = settings.BASE_DIR

    def get_example_documents(self) -> List[ExampleDocument]:
        """
        Discovers documents from:
        1. data/precomputed/ (user-provided PDF + JSON pairs)
        2. data/uploads/ (user uploaded files)
        3. data/hil_sessions/ (saved sessions)
        Strictly guarantees that any returned document has an existing PDF file on disk.
        """
        docs: List[ExampleDocument] = []

        # 1. Discover precomputed documents in data/precomputed/
        if self.precomputed_dir.exists():
            # Check standalone PDFs in precomputed_dir
            for pdf_file in sorted(self.precomputed_dir.glob("*.pdf")):
                stem = pdf_file.stem
                json_file = self.precomputed_dir / f"{stem}.json"
                try:
                    doc_info = fitz.open(str(pdf_file))
                    page_count = len(doc_info)
                except Exception:
                    page_count = 1

                docs.append(ExampleDocument(
                    id=f"precomputed_{stem}",
                    title=f"{stem}",
                    input_type=InputType.GAZETTE_BILINGUAL,
                    description=f"Precomputed bilingual corpus ({page_count} pages).",
                    pdf_path=str(pdf_file),
                    pages=page_count,
                    has_precomputed=json_file.exists()
                ))

            # Check subdirectories in precomputed_dir
            for folder in sorted(self.precomputed_dir.iterdir()):
                if folder.is_dir():
                    pdfs = [p for p in sorted(folder.glob("*.pdf")) if p.is_file()]
                    jsons = [j for j in sorted(folder.glob("*.json")) if j.is_file()]
                    if len(pdfs) == 1:
                        try:
                            doc_info = fitz.open(str(pdfs[0]))
                            page_count = len(doc_info)
                        except Exception:
                            page_count = 1

                        docs.append(ExampleDocument(
                            id=f"precomputed_{folder.name}",
                            title=f"{folder.name}",
                            input_type=InputType.GAZETTE_BILINGUAL,
                            description=f"Precomputed bilingual document ({page_count} pages).",
                            pdf_path=str(pdfs[0]),
                            pages=page_count,
                            has_precomputed=len(jsons) > 0
                        ))
                    elif len(pdfs) >= 2:
                        try:
                            doc_info = fitz.open(str(pdfs[0]))
                            page_count = len(doc_info)
                        except Exception:
                            page_count = 1

                        docs.append(ExampleDocument(
                            id=f"precomputed_{folder.name}",
                            title=f"{folder.name} (Dual Codebooks)",
                            input_type=InputType.CODEBOOK_DUAL,
                            description=f"Precomputed dual codebooks: {pdfs[0].name} & {pdfs[1].name}.",
                            pdf_path=str(pdfs[0]),
                            pdf_path_eng=str(pdfs[1]),
                            pages=page_count,
                            has_precomputed=len(jsons) > 0
                        ))

        # 2. Discover uploaded documents in data/uploads/
        if self.uploads_dir.exists():
            for folder in sorted(self.uploads_dir.iterdir()):
                if folder.is_dir():
                    pdfs = [p for p in folder.glob("*.pdf") if p.is_file()]
                    if len(pdfs) == 1:
                        try:
                            doc_info = fitz.open(str(pdfs[0]))
                            page_count = len(doc_info)
                        except Exception:
                            page_count = 1

                        docs.append(ExampleDocument(
                            id=f"upload_{folder.name}",
                            title=f"{pdfs[0].stem}",
                            input_type=InputType.GAZETTE_BILINGUAL,
                            description=f"Uploaded bilingual document ({page_count} pages).",
                            pdf_path=str(pdfs[0]),
                            pages=page_count,
                            has_precomputed=False
                        ))
                    elif len(pdfs) >= 2:
                        try:
                            doc_info = fitz.open(str(pdfs[0]))
                            page_count = len(doc_info)
                        except Exception:
                            page_count = 1

                        docs.append(ExampleDocument(
                            id=f"upload_{folder.name}",
                            title=f"{folder.name} (Dual Codebooks)",
                            input_type=InputType.CODEBOOK_DUAL,
                            description=f"Uploaded dual codebooks: {pdfs[0].name} & {pdfs[1].name}.",
                            pdf_path=str(pdfs[0]),
                            pdf_path_eng=str(pdfs[1]),
                            pages=page_count,
                            has_precomputed=False
                        ))

        # 3. Check saved sessions in data/hil_sessions/ only if the PDF exists on disk
        if self.sessions_dir.exists():
            for s_file in sorted(self.sessions_dir.glob("*.json")):
                try:
                    with open(s_file, "r", encoding="utf-8") as f:
                        s_data = json.load(f)
                    s_id = s_data.get("session_id")
                    title = s_data.get("document_id") or s_file.stem
                    m_type = s_data.get("input_type", "gazette_bilingual")
                    
                    pdf_p = s_data.get("metadata", {}).get("pdf_path")
                    if not pdf_p or not Path(pdf_p).exists():
                        pdf_p = s_data.get("metadata", {}).get("pdf_path_amh")

                    # Strictly only list session if the underlying PDF still exists on disk
                    if pdf_p and Path(pdf_p).exists() and s_id:
                        session_doc = ExampleDocument(
                            id=s_id,
                            title=title,
                            input_type=InputType(m_type) if m_type in [t.value for t in InputType] else InputType.GAZETTE_BILINGUAL,
                            description=f"Verified session with {len(s_data.get('pairs', []))} blocks.",
                            pdf_path=str(pdf_p),
                            pdf_path_eng=s_data.get("metadata", {}).get("pdf_path_eng"),
                            pages=s_data.get("total_pages", 1),
                            has_precomputed=True
                        )
                        existing_idx = next((i for i, d in enumerate(docs) if d.title == title or d.pdf_path == str(pdf_p)), None)
                        if existing_idx is not None:
                            docs[existing_idx] = session_doc
                        else:
                            docs.append(session_doc)
                except Exception:
                    pass

        return docs

    def _parse_raw_data_to_pairs(self, raw_data: Any) -> List[AlignedPair]:
        """
        Flexibly parses various JSON formats (list of objects, blocks, pairs, aligned_pairs)
        into normalized AlignedPair models.
        """
        items: List[Dict[str, Any]] = []
        if isinstance(raw_data, list):
            items = raw_data
        elif isinstance(raw_data, dict):
            # Check for w2/data style: "pages": { "1": [...], "2": [...] }
            if "pages" in raw_data and isinstance(raw_data["pages"], dict):
                for page_str, page_blocks in raw_data["pages"].items():
                    try:
                        p_num = int(page_str)
                    except Exception:
                        p_num = 1
                    if isinstance(page_blocks, list):
                        for b in page_blocks:
                            if isinstance(b, dict):
                                b_copy = dict(b)
                                b_copy.setdefault("page_number", p_num)
                                items.append(b_copy)

            # Check for "pages": [ { "page_number": 1, "blocks": [...] } ]
            elif "pages" in raw_data and isinstance(raw_data["pages"], list):
                for page_obj in raw_data["pages"]:
                    if isinstance(page_obj, dict):
                        p_num = page_obj.get("page_number") or page_obj.get("page") or 1
                        b_list = page_obj.get("blocks") or page_obj.get("pairs") or []
                        for b in b_list:
                            if isinstance(b, dict):
                                b_copy = dict(b)
                                b_copy.setdefault("page_number", p_num)
                                items.append(b_copy)

            elif "blocks" in raw_data and isinstance(raw_data["blocks"], list):
                items = raw_data["blocks"]
            elif "pairs" in raw_data and isinstance(raw_data["pairs"], list):
                items = raw_data["pairs"]
            elif "aligned_pairs" in raw_data and isinstance(raw_data["aligned_pairs"], list):
                items = raw_data["aligned_pairs"]
            elif "data" in raw_data and isinstance(raw_data["data"], list):
                items = raw_data["data"]

        pairs: List[AlignedPair] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue

            amh = str(item.get("amharic") or item.get("amh") or item.get("amharic_text") or item.get("amh_text") or "").strip()
            eng = str(item.get("english") or item.get("eng") or item.get("english_text") or item.get("eng_text") or "").strip()
            
            art_no = item.get("article_number") or item.get("article") or item.get("art_no") or item.get("art")
            art_str = str(art_no).strip() if art_no is not None else None
            if not art_str:
                art_match = re.search(r"(?:Article|Art\.)\s*(\d+)", eng, re.IGNORECASE)
                if art_match:
                    art_str = art_match.group(1)
                else:
                    amh_art_match = re.search(r"አንቀጽ\s*([፩-፱፲፳፴፵፶፷፸፹፺፻\d]+)", amh)
                    if amh_art_match:
                        art_str = amh_art_match.group(1)

            raw_type = str(item.get("type") or item.get("block_type") or "").lower()
            if "head" in raw_type or "federal negarit" in eng.lower() or "ነጋሪት ጋዜጣ" in amh:
                b_type = BlockType.HEADER
            elif "tit" in raw_type:
                b_type = BlockType.TITLE
            elif "toc" in raw_type or "content" in eng.lower() or "ማውጫ" in amh:
                b_type = BlockType.TOC
            elif "sub" in raw_type:
                b_type = BlockType.SUB_ARTICLE
            elif "art" in raw_type or art_str or "አጭር ርዕስ" in amh or "short title" in eng.lower():
                b_type = BlockType.ARTICLE
            else:
                b_type = BlockType.PARAGRAPH

            line_id = item.get("line_id") or item.get("id") or idx
            try:
                line_id = int(line_id)
            except Exception:
                line_id = idx

            page_num = item.get("page_number") or item.get("page") or 1
            try:
                page_num = int(page_num)
            except Exception:
                page_num = 1

            conf = item.get("confidence") or 0.95
            try:
                conf = float(conf)
            except Exception:
                conf = 0.95

            status_str = str(item.get("status") or "pending").lower()
            if status_str == "verified": p_status = HILStatus.VERIFIED
            elif status_str == "flagged": p_status = HILStatus.FLAGGED
            elif status_str == "modified": p_status = HILStatus.MODIFIED
            else: p_status = HILStatus.PENDING

            pairs.append(AlignedPair(
                id=str(item.get("id") or f"pair_{idx}"),
                line_id=line_id,
                article_number=art_str,
                type=b_type,
                page_number=page_num,
                amharic=amh,
                english=eng,
                confidence=conf,
                status=p_status,
                notes=item.get("notes")
            ))

        return pairs

    def load_precomputed_data(self, example_id: str) -> Optional[List[AlignedPair]]:
        """
        Loads precomputed data from data/precomputed/, precomputed/, or data/hil_sessions/.
        """
        candidate_dirs = [self.precomputed_dir, self.base_dir / "precomputed"]

        # 1. Check precomputed directories
        key = example_id.replace("precomputed_", "", 1) if example_id.startswith("precomputed_") else example_id
        for p_dir in candidate_dirs:
            if not p_dir.exists():
                continue

            # Direct file check: <key>.json
            direct_json = p_dir / f"{key}.json"
            if direct_json.exists():
                try:
                    with open(direct_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    parsed = self._parse_raw_data_to_pairs(data)
                    if parsed:
                        return parsed
                except Exception as e:
                    logger.warning(f"Error loading precomputed JSON {direct_json}: {e}")

            # Subdirectory check: <key>/<*.json>
            folder = p_dir / key
            if folder.exists() and folder.is_dir():
                jsons = list(folder.glob("*.json"))
                if jsons:
                    try:
                        with open(jsons[0], "r", encoding="utf-8") as f:
                            data = json.load(f)
                        parsed = self._parse_raw_data_to_pairs(data)
                        if parsed:
                            return parsed
                    except Exception as e:
                        logger.warning(f"Error loading precomputed JSON {jsons[0]}: {e}")

            # Fuzzy match any json matching the stem
            for j_file in p_dir.glob("*.json"):
                if j_file.stem.lower() == key.lower():
                    try:
                        with open(j_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        parsed = self._parse_raw_data_to_pairs(data)
                        if parsed:
                            return parsed
                    except Exception as e:
                        pass

        # 2. Check saved sessions in data/hil_sessions/
        session_file = self.sessions_dir / f"{example_id}.json"
        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self._parse_raw_data_to_pairs(data)
            except Exception as e:
                logger.warning(f"Error loading session {example_id}: {e}")

        return None

    def save_session(self, session: HILSession):
        path = self.sessions_dir / f"{session.session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(session.model_dump_json(indent=2))

    def load_session(self, session_id: str) -> Optional[HILSession]:
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return HILSession(**data)

    def export_corpus_json(self, session: HILSession, only_verified: bool = False) -> str:
        filtered = session.pairs
        if only_verified:
            filtered = [p for p in session.pairs if p.status in (HILStatus.VERIFIED, HILStatus.MODIFIED)]

        export_data = {
            "document_id": session.document_id,
            "session_id": session.session_id,
            "exported_at": datetime.utcnow().isoformat(),
            "total_pairs": len(filtered),
            "audit_summary": session.audit.model_dump(),
            "corpus": [
                {
                    "line_id": p.line_id,
                    "type": p.type.value,
                    "amharic": p.amharic,
                    "english": p.english,
                    "confidence": p.confidence,
                    "hil_status": p.status.value,
                    "page": p.page_number
                }
                for p in filtered
            ]
        }
        filename = f"corpus_{session.document_id}_{session.session_id[:8]}.json"
        out_path = self.exports_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return str(out_path)

    def export_corpus_csv(self, session: HILSession, only_verified: bool = False) -> str:
        filtered = session.pairs
        if only_verified:
            filtered = [p for p in session.pairs if p.status in (HILStatus.VERIFIED, HILStatus.MODIFIED)]

        filename = f"corpus_{session.document_id}_{session.session_id[:8]}.csv"
        out_path = self.exports_dir / filename

        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["line_id", "type", "amharic", "english", "confidence", "status", "page"])
            for p in filtered:
                writer.writerow([p.line_id, p.type.value, p.amharic, p.english, p.confidence, p.status.value, p.page_number])

        return str(out_path)

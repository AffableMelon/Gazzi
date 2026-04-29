import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Response
from fastapi.responses import FileResponse

from app.config import settings
from app.models import (
    InputType, HILStatus, PipelineRunRequest, HILUpdatePairRequest,
    HILBatchApproveRequest, HILSession, ExampleDocument, AuditSummary
)
from app.services.llm_service import LLMService
from app.services.storage_service import StorageService
from app.services.gazette_pipeline import GazettePipeline
from app.services.dual_code_pipeline import DualCodePipeline
from app.services.pdf_extractor import PDFExtractor
from app.services.audit_service import AuditService

router = APIRouter(prefix="/api")
storage = StorageService()
pdf_extractor = PDFExtractor(dpi=150)

@router.get("/health")
def get_health():
    llm = LLMService()
    return {
        "status": "online",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_configured": llm.is_configured(),
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "openai_configured": bool(settings.OPENAI_API_KEY),
        "fallback_enabled": settings.FALLBACK_TO_LOCAL_OCR,
        "env_path": str(settings.BASE_DIR / ".env")
    }

@router.get("/examples", response_model=List[ExampleDocument])
def list_examples():
    """Lists preloaded real Ethiopian legal documents from w1, w2, and civils."""
    return storage.get_example_documents()

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    file_eng: Optional[UploadFile] = File(None),
    json_file: Optional[UploadFile] = File(None),
    json_text: Optional[str] = Form(None)
):
    """Uploads user PDF(s) to process, with optional Gemini Chat extracted JSON."""
    upload_id = str(uuid.uuid4())[:8]
    dest_dir = settings.UPLOADS_DIR / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / file.filename
    with open(dest_file, "wb") as f:
        shutil.copyfileobj(file.file, f)

    dest_eng_file = None
    if file_eng:
        dest_eng_file = dest_dir / file_eng.filename
        with open(dest_eng_file, "wb") as f:
            shutil.copyfileobj(file_eng.file, f)

    has_json = False
    if json_file and json_file.filename:
        dest_json = dest_dir / "extraction.json"
        with open(dest_json, "wb") as f:
            shutil.copyfileobj(json_file.file, f)
        has_json = True
    elif json_text and json_text.strip():
        cleaned = json_text.strip()
        # Automatically strip markdown fences from Gemini chat
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        
        dest_json = dest_dir / "extraction.json"
        with open(dest_json, "w", encoding="utf-8") as f:
            f.write(cleaned)
        has_json = True

    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "file_path": str(dest_file),
        "file_path_eng": str(dest_eng_file) if dest_eng_file else None,
        "has_json": has_json
    }

@router.post("/pipeline/run", response_model=HILSession)
def run_pipeline(req: PipelineRunRequest):
    """
    Executes the legal AI extraction and alignment pipeline.
    Supports:
    - Type 1: Single bilingual gazette PDF
    - Type 2: Dual code books (Amharic + English PDFs)
    """
    llm = LLMService(provider=req.llm_provider, api_key=req.api_key)

    pdf_path = None
    pdf_path_eng = None
    doc_title = None

    # 1. Resolve file path from examples or uploads
    if req.example_id:
        existing_session = storage.load_session(req.example_id)
        if existing_session:
            return existing_session

        examples = {e.id: e for e in storage.get_example_documents()}
        if req.example_id not in examples:
            raise HTTPException(status_code=404, detail="Example not found")
        ex = examples[req.example_id]
        pdf_path = ex.pdf_path
        pdf_path_eng = ex.pdf_path_eng
        doc_title = ex.title
        req.input_type = ex.input_type

    elif req.uploaded_file_id:
        match = list(settings.UPLOADS_DIR.glob(f"{req.uploaded_file_id}/*"))
        if not match:
            raise HTTPException(status_code=404, detail="Uploaded file not found")
        
        pdf_files = [str(f) for f in match if f.suffix.lower() == ".pdf"]
        if not pdf_files:
            raise HTTPException(status_code=400, detail="No PDF files found in upload")
        
        pdf_path = pdf_files[0]
        if len(pdf_files) >= 2:
            pdf_path_eng = pdf_files[1]
        elif req.uploaded_file_id_eng:
            match_eng = list(settings.UPLOADS_DIR.glob(f"{req.uploaded_file_id_eng}/*"))
            eng_pdfs = [str(f) for f in match_eng if f.suffix.lower() == ".pdf"]
            if eng_pdfs:
                pdf_path_eng = eng_pdfs[0]

        # If user uploaded or pasted a JSON (e.g. from Gemini Web Chat), parse and return it immediately
        json_files = [f for f in match if f.suffix.lower() == ".json"]
        if json_files:
            try:
                with open(json_files[0], "r", encoding="utf-8") as jf:
                    raw_json = json.load(jf)
                pairs = storage._parse_raw_data_to_pairs(raw_json)
                if pairs:
                    doc_info = pdf_extractor.get_document_info(pdf_path)
                    total_pages = doc_info["page_count"]
                    audit = AuditService.audit_pairs(pairs)
                    session_id = str(uuid.uuid4())
                    doc_title = Path(pdf_path).stem
                    session = HILSession(
                        session_id=session_id,
                        document_id=doc_title,
                        input_type=req.input_type,
                        created_at=datetime.utcnow().isoformat(),
                        updated_at=datetime.utcnow().isoformat(),
                        file_name=Path(pdf_path).name,
                        total_pages=total_pages,
                        pages_processed=list(range(1, total_pages + 1)),
                        pairs=pairs,
                        audit=audit,
                        metadata={
                            "source_pdf": Path(pdf_path).name,
                            "pdf_path": pdf_path,
                            "pdf_path_eng": pdf_path_eng,
                            "extractor_mode": "Gemini Chat JSON Import",
                            "provider": "gemini_chat_web"
                        }
                    )
                    storage.save_session(session)
                    return session
            except Exception as e:
                pass
    else:
        examples = storage.get_example_documents()
        if examples:
            ex = examples[0]
            pdf_path = ex.pdf_path
            doc_title = ex.title
            req.input_type = ex.input_type
        else:
            raise HTTPException(status_code=400, detail="No documents available. Please upload a PDF using the Upload button.")

    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(status_code=400, detail=f"PDF file not found: {pdf_path}")

    # 2. Execute according to Input Type
    if req.input_type == InputType.GAZETTE_BILINGUAL:
        gazette_pipe = GazettePipeline(llm_service=llm)
        session = gazette_pipe.process(
            pdf_path=pdf_path,
            page_start=req.page_start,
            page_end=req.page_end,
            document_id=doc_title or Path(pdf_path).stem
        )
        return session

    elif req.input_type == InputType.CODEBOOK_DUAL:
        if not pdf_path_eng or not Path(pdf_path_eng).exists():
            ex_dual = [e for e in storage.get_example_documents() if e.input_type == InputType.CODEBOOK_DUAL]
            if ex_dual and ex_dual[0].pdf_path_eng:
                pdf_path_eng = ex_dual[0].pdf_path_eng
            else:
                pdf_path_eng = pdf_path  # fallback to same file if only 1 uploaded

        dual_pipe = DualCodePipeline(llm_service=llm)
        session = dual_pipe.process(
            pdf_path_amh=pdf_path,
            pdf_path_eng=pdf_path_eng,
            page_start=req.page_start,
            page_end=req.page_end,
            document_id=doc_title or f"{Path(pdf_path).stem}_dual"
        )
        return session

    else:
        raise HTTPException(status_code=400, detail="Invalid input type")

@router.get("/session/{session_id}", response_model=HILSession)
def get_session(session_id: str):
    session = storage.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.post("/session/{session_id}/update-pair")
def update_pair(session_id: str, req: HILUpdatePairRequest):
    """In-place human edit of an extracted bilingual block."""
    session = storage.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    target = None
    for p in session.pairs:
        if p.id == req.pair_id:
            target = p
            break

    if not target:
        raise HTTPException(status_code=404, detail="Pair not found")

    if req.amharic is not None:
        target.amharic = req.amharic
    if req.english is not None:
        target.english = req.english
    if req.notes is not None:
        target.notes = req.notes
    if req.article_number is not None:
        target.article_number = req.article_number
    if req.type is not None:
        target.type = req.type

    # If status explicitly sent, apply it; otherwise if text changed, mark as MODIFIED
    if req.status is not None:
        target.status = req.status
    else:
        target.status = HILStatus.MODIFIED

    # Re-audit session metrics
    session.audit = AuditService.audit_pairs(session.pairs)
    storage.save_session(session)

    return {"status": "success", "pair": target, "audit": session.audit}

@router.post("/session/{session_id}/batch-approve")
def batch_approve(session_id: str, req: HILBatchApproveRequest):
    """Batch marks pairs as VERIFIED by human annotator."""
    session = storage.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    target_set = set(req.pair_ids)
    for p in session.pairs:
        if not target_set or p.id in target_set:
            p.status = HILStatus.VERIFIED

    session.audit = AuditService.audit_pairs(session.pairs)
    storage.save_session(session)

    return {"status": "success", "verified_count": session.audit.verified_count, "audit": session.audit}

@router.get("/session/{session_id}/page-image/{page_number}")
def get_page_image(session_id: str, page_number: int, lang: Optional[str] = Query("amh")):
    """Returns rendered page image for visual verification in the dashboard."""
    session = storage.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    pdf_path = None
    if lang == "eng" and "pdf_path_eng" in session.metadata:
        pdf_path = session.metadata["pdf_path_eng"]
    elif "pdf_path" in session.metadata:
        pdf_path = session.metadata["pdf_path"]
    elif "pdf_path_amh" in session.metadata:
        pdf_path = session.metadata["pdf_path_amh"]

    if not pdf_path or not Path(pdf_path).exists():
        src = session.metadata.get("source_pdf") or session.file_name.split(" + ")[0]
        matches = list(settings.BASE_DIR.glob(f"**/{src}"))
        if matches:
            pdf_path = str(matches[0])

    if not pdf_path or not Path(pdf_path).exists():
        examples = storage.get_example_documents()
        if examples:
            pdf_path = examples[0].pdf_path
        else:
            raise HTTPException(status_code=404, detail="Source PDF file not found on disk")

    try:
        img_bytes = pdf_extractor.render_page_to_png(pdf_path, page_number, dpi=140)
        return Response(content=img_bytes, media_type="image/png")
    except Exception:
        # Fallback to page 1 if requested page is out of bounds
        img_bytes = pdf_extractor.render_page_to_png(pdf_path, 1, dpi=140)
        return Response(content=img_bytes, media_type="image/png")

@router.get("/session/{session_id}/export/{export_format}")
def export_corpus(
    session_id: str,
    export_format: str,
    only_verified: bool = Query(False)
):
    """Downloads the verified bilingual parallel corpus in JSON or CSV."""
    session = storage.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    fmt = export_format.lower()
    if fmt == "json":
        filepath = storage.export_corpus_json(session, only_verified=only_verified)
        return FileResponse(filepath, media_type="application/json", filename=Path(filepath).name)
    elif fmt == "csv":
        filepath = storage.export_corpus_csv(session, only_verified=only_verified)
        return FileResponse(filepath, media_type="text/csv", filename=Path(filepath).name)
    else:
        raise HTTPException(status_code=400, detail="Supported export formats: 'json', 'csv'")

@router.get("/prompts")
def get_prompts():
    """Returns official extraction and alignment prompts to copy into Gemini web chat."""
    from app.services.llm_service import GAZETTE_EXTRACTION_PROMPT, BILINGUAL_ALIGNMENT_PROMPT
    return {
        "gazette_prompt": GAZETTE_EXTRACTION_PROMPT,
        "codebook_prompt": BILINGUAL_ALIGNMENT_PROMPT
    }

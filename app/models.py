from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class InputType(str, Enum):
    GAZETTE_BILINGUAL = "gazette_bilingual"  # Type 1: Single PDF with 2-column bilingual layout
    CODEBOOK_DUAL = "codebook_dual"          # Type 2: Dual PDFs (Amharic PDF + English PDF)

class HILStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FLAGGED = "flagged"
    MODIFIED = "modified"
    UNALIGNED = "unaligned"

class BlockType(str, Enum):
    HEADER = "header"
    TITLE = "title"
    PREAMBLE = "preamble"
    ARTICLE = "article"
    SUB_ARTICLE = "sub_article"
    PARAGRAPH = "paragraph"
    TOC = "toc"
    TABLE = "table"
    UNKNOWN = "unknown"

class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

class AlignedPair(BaseModel):
    id: str
    line_id: int
    article_number: Optional[str] = None
    type: BlockType = BlockType.PARAGRAPH
    page_number: int = 1
    amharic: str = ""
    english: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
    status: HILStatus = HILStatus.PENDING
    notes: Optional[str] = None
    discrepancy_reason: Optional[str] = None
    amharic_bbox: Optional[BoundingBox] = None
    english_bbox: Optional[BoundingBox] = None

class DiscrepancyReport(BaseModel):
    pair_id: str
    line_id: int
    reason: str
    score: float
    amharic_snippet: str
    english_snippet: str

class AuditSummary(BaseModel):
    total_pairs: int = 0
    verified_count: int = 0
    pending_count: int = 0
    flagged_count: int = 0
    modified_count: int = 0
    match_accuracy_pct: float = 100.0
    missing_amharic_count: int = 0
    missing_english_count: int = 0
    line_count_mismatches: int = 0
    potential_hallucinations: int = 0
    discrepancies: List[DiscrepancyReport] = []

class HILSession(BaseModel):
    session_id: str
    document_id: str
    input_type: InputType
    created_at: str
    updated_at: str
    file_name: str
    total_pages: int
    pages_processed: List[int]
    pairs: List[AlignedPair]
    audit: AuditSummary
    metadata: Dict[str, Any] = {}

class PipelineRunRequest(BaseModel):
    input_type: InputType = InputType.GAZETTE_BILINGUAL
    example_id: Optional[str] = None
    uploaded_file_id: Optional[str] = None
    uploaded_file_id_eng: Optional[str] = None
    page_start: int = 1
    page_end: int = 3
    llm_provider: Optional[str] = None  # "gemini" or "openai"
    api_key: Optional[str] = None       # optional manual override for .env
    custom_prompt: Optional[str] = None

class HILUpdatePairRequest(BaseModel):
    session_id: str
    pair_id: str
    amharic: Optional[str] = None
    english: Optional[str] = None
    status: Optional[HILStatus] = None
    notes: Optional[str] = None
    article_number: Optional[str] = None
    type: Optional[BlockType] = None

class HILBatchApproveRequest(BaseModel):
    session_id: str
    pair_ids: List[str]

class ExampleDocument(BaseModel):
    id: str
    title: str
    input_type: InputType
    description: str
    pdf_path: str
    pdf_path_eng: Optional[str] = None
    pages: int
    has_precomputed: bool

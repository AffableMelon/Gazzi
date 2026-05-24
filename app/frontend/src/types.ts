export type InputType = 'gazette_bilingual' | 'codebook_dual';

export type HILStatus = 'pending' | 'verified' | 'flagged' | 'modified' | 'unaligned';

export type BlockType = 
  | 'header' 
  | 'title' 
  | 'preamble' 
  | 'article' 
  | 'sub_article' 
  | 'paragraph' 
  | 'toc' 
  | 'table' 
  | 'unknown';

export interface BoundingBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface AlignedPair {
  id: string;
  line_id: number;
  article_number?: string | null;
  type: BlockType;
  page_number: number;
  amharic: string;
  english: string;
  confidence: number;
  status: HILStatus;
  notes?: string | null;
  discrepancy_reason?: string | null;
  amharic_bbox?: BoundingBox | null;
  english_bbox?: BoundingBox | null;
}

export interface DiscrepancyReport {
  pair_id: string;
  line_id: number;
  reason: string;
  score: number;
  amharic_snippet: string;
  english_snippet: string;
}

export interface AuditSummary {
  total_pairs: number;
  verified_count: number;
  pending_count: number;
  flagged_count: number;
  modified_count: number;
  match_accuracy_pct: number;
  missing_amharic_count: number;
  missing_english_count: number;
  line_count_mismatches: number;
  potential_hallucinations: number;
  discrepancies: DiscrepancyReport[];
}

export interface HILSession {
  session_id: string;
  document_id: string;
  input_type: InputType;
  created_at: string;
  updated_at: string;
  file_name: string;
  total_pages: number;
  pages_processed: number[];
  pairs: AlignedPair[];
  audit: AuditSummary;
  metadata: Record<string, any>;
}

export interface ExampleDocument {
  id: string;
  title: string;
  input_type: InputType;
  description: string;
  pdf_path: string;
  pdf_path_eng?: string | null;
  pages: number;
  has_precomputed: boolean;
}

export interface HealthStatus {
  status: string;
  llm_provider: string;
  llm_configured: boolean;
  gemini_configured: boolean;
  openai_configured: boolean;
  fallback_enabled: boolean;
  env_path: string;
}

export interface PipelineRunRequest {
  input_type: InputType;
  example_id?: string;
  uploaded_file_id?: string;
  uploaded_file_id_eng?: string;
  page_start: number;
  page_end: number;
  llm_provider?: string;
  api_key?: string;
  custom_prompt?: string;
}

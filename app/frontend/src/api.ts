import { 
  HealthStatus, ExampleDocument, HILSession, 
  PipelineRunRequest, AlignedPair, AuditSummary 
} from './types';

const API_BASE = '/api';

export const api = {
  async getHealth(): Promise<HealthStatus> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to fetch health status');
    return res.json();
  },

  async getExamples(): Promise<ExampleDocument[]> {
    const res = await fetch(`${API_BASE}/examples`);
    if (!res.ok) throw new Error('Failed to fetch examples');
    return res.json();
  },

  async uploadPdf(
    file: File, 
    fileEng?: File, 
    jsonText?: string, 
    jsonFile?: File
  ): Promise<{ upload_id: string; filename: string; file_path: string; file_path_eng?: string; has_json: boolean }> {
    const formData = new FormData();
    formData.append('file', file);
    if (fileEng) formData.append('file_eng', fileEng);
    if (jsonFile) formData.append('json_file', jsonFile);
    if (jsonText && jsonText.trim()) formData.append('json_text', jsonText.trim());

    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  async getPrompts(): Promise<{ gazette_prompt: string; codebook_prompt: string }> {
    const res = await fetch(`${API_BASE}/prompts`);
    if (!res.ok) throw new Error('Failed to fetch prompts');
    return res.json();
  },

  async runPipeline(req: PipelineRunRequest): Promise<HILSession> {
    const res = await fetch(`${API_BASE}/pipeline/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Pipeline execution failed');
    }
    return res.json();
  },

  async getSession(sessionId: string): Promise<HILSession> {
    const res = await fetch(`${API_BASE}/session/${sessionId}`);
    if (!res.ok) throw new Error('Session not found');
    return res.json();
  },

  async updatePair(
    sessionId: string, 
    pairId: string, 
    updates: Partial<AlignedPair>
  ): Promise<{ status: string; pair: AlignedPair; audit: AuditSummary }> {
    const res = await fetch(`${API_BASE}/session/${sessionId}/update-pair`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        pair_id: pairId,
        ...updates
      }),
    });
    if (!res.ok) throw new Error('Failed to update pair');
    return res.json();
  },

  async batchApprove(sessionId: string, pairIds: string[]): Promise<{ status: string; verified_count: number; audit: AuditSummary }> {
    const res = await fetch(`${API_BASE}/session/${sessionId}/batch-approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        pair_ids: pairIds
      }),
    });
    if (!res.ok) throw new Error('Batch approval failed');
    return res.json();
  },

  getPageImageUrl(sessionId: string, pageNumber: number, lang: 'amh' | 'eng' = 'amh'): string {
    return `${API_BASE}/session/${sessionId}/page-image/${pageNumber}?lang=${lang}&t=${Date.now()}`;
  },

  getExportUrl(sessionId: string, format: 'json' | 'csv', onlyVerified: boolean = false): string {
    return `${API_BASE}/session/${sessionId}/export/${format}?only_verified=${onlyVerified}`;
  }
};

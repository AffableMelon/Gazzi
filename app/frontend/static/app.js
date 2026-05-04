// Ethiopian Legal AI Human-in-the-Loop Client
const API_BASE = '/api';

const api = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to fetch health status');
    return res.json();
  },

  async getExamples() {
    const res = await fetch(`${API_BASE}/examples`);
    if (!res.ok) throw new Error('Failed to fetch examples');
    return res.json();
  },

  async runPipeline(req) {
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

  async getSession(sessionId) {
    const res = await fetch(`${API_BASE}/session/${sessionId}`);
    if (!res.ok) throw new Error('Session not found');
    return res.json();
  },

  async updatePair(sessionId, pairId, updates) {
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

  async batchApprove(sessionId, pairIds) {
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

  getPageImageUrl(sessionId, pageNumber, lang = 'amh') {
    return `${API_BASE}/session/${sessionId}/page-image/${pageNumber}?lang=${lang}&t=${Date.now()}`;
  },

  getExportUrl(sessionId, format, onlyVerified = false) {
    return `${API_BASE}/session/${sessionId}/export/${format}?only_verified=${onlyVerified}`;
  }
};

class LegalHILApp {
  constructor() {
    this.currentSession = null;
    this.examples = [];
    this.selectedInputType = 'gazette_bilingual';
    this.currentFilter = 'all';
    this.searchQuery = '';
    this.activePageNum = 1;
    this.activePdfLang = 'amh';
    this.activePairId = null;
    this.init();
  }

  async init() {
    this.bindEvents();
    await this.loadExamples();
  }

  async loadExamples() {
    try {
      this.examples = await api.getExamples();
      this.renderExampleDropdown();
      const initialDoc = this.examples.find(e => e.input_type === this.selectedInputType);
      if (initialDoc) {
        await this.loadDocument(initialDoc.id);
      }
    } catch (e) {
      this.showToast('Could not load documents', 'error');
    }
  }

  renderExampleDropdown() {
    const select = document.getElementById('example-select');
    if (!select) return;

    const filtered = this.examples.filter(e => e.input_type === this.selectedInputType);
    select.innerHTML = filtered.map(ex => `
      <option value="${ex.id}">${ex.title} (${ex.pages} pages)</option>
    `).join('');
  }

  bindEvents() {
    const tabGazette = document.getElementById('tab-gazette');
    const tabCodebook = document.getElementById('tab-codebook');

    tabGazette?.addEventListener('click', async () => {
      if (this.selectedInputType === 'gazette_bilingual') return;
      this.selectedInputType = 'gazette_bilingual';
      this.updateModeTabUI();
      this.renderExampleDropdown();
      const first = this.examples.find(e => e.input_type === 'gazette_bilingual');
      if (first) await this.loadDocument(first.id);
    });

    tabCodebook?.addEventListener('click', async () => {
      if (this.selectedInputType === 'codebook_dual') return;
      this.selectedInputType = 'codebook_dual';
      this.updateModeTabUI();
      this.renderExampleDropdown();
      const first = this.examples.find(e => e.input_type === 'codebook_dual');
      if (first) await this.loadDocument(first.id);
    });

    const select = document.getElementById('example-select');
    select?.addEventListener('change', async (e) => {
      const docId = e.target.value;
      if (docId) await this.loadDocument(docId);
    });

    document.getElementById('btn-run-pipeline')?.addEventListener('click', async () => {
      if (select && select.value) await this.loadDocument(select.value);
    });

    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const target = e.currentTarget;
        const filter = target.getAttribute('data-filter');
        if (filter) {
          this.currentFilter = filter;
          this.updateFilterButtons();
          this.renderPairs();
        }
      });
    });

    document.getElementById('search-input')?.addEventListener('input', (e) => {
      this.searchQuery = e.target.value.toLowerCase();
      this.renderPairs();
    });

    document.getElementById('btn-batch-approve')?.addEventListener('click', () => {
      this.batchApproveAll();
    });

    document.getElementById('btn-prev-page')?.addEventListener('click', () => {
      if (!this.currentSession) return;
      if (this.activePageNum > 1) {
        this.activePageNum--;
        this.updatePdfPreview();
      }
    });

    document.getElementById('btn-next-page')?.addEventListener('click', () => {
      if (!this.currentSession) return;
      const maxPages = this.currentSession.total_pages || 10;
      if (this.activePageNum < maxPages) {
        this.activePageNum++;
        this.updatePdfPreview();
      }
    });

    const btnPdfAmh = document.getElementById('btn-pdf-amh');
    const btnPdfEng = document.getElementById('btn-pdf-eng');

    btnPdfAmh?.addEventListener('click', () => {
      this.activePdfLang = 'amh';
      btnPdfAmh.className = "px-2 py-0.5 font-semibold bg-slate-800 text-white";
      if (btnPdfEng) btnPdfEng.className = "px-2 py-0.5 text-slate-600 hover:bg-slate-100";
      this.updatePdfPreview();
    });

    btnPdfEng?.addEventListener('click', () => {
      this.activePdfLang = 'eng';
      btnPdfEng.className = "px-2 py-0.5 font-semibold bg-slate-800 text-white";
      if (btnPdfAmh) btnPdfAmh.className = "px-2 py-0.5 text-slate-600 hover:bg-slate-100";
      this.updatePdfPreview();
    });

    document.getElementById('btn-export-json')?.addEventListener('click', () => {
      if (!this.currentSession) return;
      window.open(api.getExportUrl(this.currentSession.session_id, 'json', false), '_blank');
    });

    document.getElementById('btn-export-csv')?.addEventListener('click', () => {
      if (!this.currentSession) return;
      window.open(api.getExportUrl(this.currentSession.session_id, 'csv', false), '_blank');
    });
  }

  updateModeTabUI() {
    const tabGazette = document.getElementById('tab-gazette');
    const tabCodebook = document.getElementById('tab-codebook');
    const langToggle = document.getElementById('pdf-lang-toggle');

    if (this.selectedInputType === 'gazette_bilingual') {
      tabGazette?.classList.add('bg-white', 'text-slate-900', 'shadow-2xs', 'font-semibold');
      tabGazette?.classList.remove('text-slate-600');
      tabCodebook?.classList.remove('bg-white', 'text-slate-900', 'shadow-2xs', 'font-semibold');
      tabCodebook?.classList.add('text-slate-600');
      langToggle?.classList.add('hidden');
      langToggle?.classList.remove('inline-flex');
    } else {
      tabCodebook?.classList.add('bg-white', 'text-slate-900', 'shadow-2xs', 'font-semibold');
      tabCodebook?.classList.remove('text-slate-600');
      tabGazette?.classList.remove('bg-white', 'text-slate-900', 'shadow-2xs', 'font-semibold');
      tabGazette?.classList.add('text-slate-600');
      langToggle?.classList.remove('hidden');
      langToggle?.classList.add('inline-flex');
    }
  }

  updateFilterButtons() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
      const f = btn.getAttribute('data-filter');
      if (f === this.currentFilter) {
        btn.className = "filter-btn px-2.5 py-1 rounded font-semibold bg-slate-900 text-white";
      } else {
        btn.className = "filter-btn px-2.5 py-1 rounded font-medium text-slate-600 hover:bg-slate-200";
      }
    });
  }

  async loadDocument(exampleId) {
    this.setLoading(true);
    try {
      const pageStartInput = document.getElementById('page-start');
      const pageEndInput = document.getElementById('page-end');
      const start = pageStartInput ? parseInt(pageStartInput.value, 10) || 1 : 1;
      const end = pageEndInput ? parseInt(pageEndInput.value, 10) || 5 : 5;

      const session = await api.runPipeline({
        input_type: this.selectedInputType,
        example_id: exampleId,
        page_start: start,
        page_end: end
      });

      this.currentSession = session;
      this.activePageNum = session.pages_processed[0] || 1;
      this.activePairId = session.pairs[0]?.id || null;

      this.renderMetrics();
      this.renderPairs();
      this.updatePdfPreview();
      this.showToast(`Loaded "${session.document_id}"`, 'info');
    } catch (e) {
      this.showToast(e.message || 'Failed to load document', 'error');
    } finally {
      this.setLoading(false);
    }
  }

  renderMetrics() {
    if (!this.currentSession) return;
    const a = this.currentSession.audit;

    const totalEl = document.getElementById('metric-total');
    const verifiedEl = document.getElementById('metric-verified');
    const flaggedEl = document.getElementById('metric-flagged');
    const accuracyEl = document.getElementById('metric-accuracy');
    const progressEl = document.getElementById('metric-progress-bar');
    const pctText = document.getElementById('metric-verified-pct');

    if (totalEl) totalEl.textContent = a.total_pairs.toString();
    if (verifiedEl) verifiedEl.textContent = `${a.verified_count + a.modified_count}`;
    if (flaggedEl) flaggedEl.textContent = a.flagged_count.toString();
    if (accuracyEl) accuracyEl.textContent = `${a.match_accuracy_pct}%`;

    const pct = a.total_pairs > 0 
      ? Math.round(((a.verified_count + a.modified_count) / a.total_pairs) * 100) 
      : 0;

    if (progressEl) progressEl.style.width = `${pct}%`;
    if (pctText) pctText.textContent = `${pct}%`;
  }

  renderPairs() {
    const container = document.getElementById('pairs-container');
    if (!container || !this.currentSession) return;

    let list = this.currentSession.pairs;

    if (this.currentFilter === 'flagged') {
      list = list.filter(p => p.status === 'flagged');
    } else if (this.currentFilter === 'verified') {
      list = list.filter(p => p.status === 'verified' || p.status === 'modified');
    } else if (this.currentFilter === 'pending') {
      list = list.filter(p => p.status === 'pending');
    }

    if (this.searchQuery) {
      list = list.filter(p => 
        p.amharic.toLowerCase().includes(this.searchQuery) ||
        p.english.toLowerCase().includes(this.searchQuery) ||
        (p.article_number && p.article_number.includes(this.searchQuery)) ||
        p.line_id.toString().includes(this.searchQuery)
      );
    }

    if (list.length === 0) {
      container.innerHTML = `
        <div class="text-center py-12 text-slate-400 bg-white rounded-lg border border-dashed border-slate-300 text-xs">
          No aligned blocks match the current filter.
        </div>
      `;
      return;
    }

    container.innerHTML = list.map(pair => this.renderPairCard(pair)).join('');
    this.bindPairCardEvents();
  }

  renderPairCard(pair) {
    const isVerified = pair.status === 'verified';
    const isFlagged = pair.status === 'flagged';
    const isModified = pair.status === 'modified';
    const isActive = pair.id === this.activePairId;

    let statusBadge = `
      <span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600 border border-slate-200">
        Pending
      </span>
    `;

    if (isVerified) {
      statusBadge = `
        <span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
          Verified
        </span>
      `;
    } else if (isModified) {
      statusBadge = `
        <span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-sky-50 text-sky-700 border border-sky-200">
          Edited
        </span>
      `;
    } else if (isFlagged) {
      statusBadge = `
        <span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-50 text-amber-700 border border-amber-300">
          Review Needed
        </span>
      `;
    }

    const discrepancyAlert = pair.discrepancy_reason ? `
      <div class="mb-2 px-2.5 py-1 rounded bg-amber-50 border border-amber-200 text-[11px] text-amber-800 flex items-center justify-between">
        <span><strong>Notice:</strong> ${pair.discrepancy_reason}</span>
        <span class="text-[10px] font-mono opacity-80">Score: ${(pair.confidence * 100).toFixed(0)}%</span>
      </div>
    ` : '';

    return `
      <div 
        class="pair-card bg-white rounded-lg border ${isActive ? 'border-slate-800 ring-2 ring-slate-900/10' : isFlagged ? 'border-amber-300' : 'border-slate-200'} shadow-2xs hover:border-slate-400 transition p-3.5 mb-3 cursor-pointer" 
        data-pair-id="${pair.id}"
        data-page="${pair.page_number}"
      >
        ${discrepancyAlert}
        
        <div class="flex items-center justify-between border-b border-slate-100 pb-2 mb-2.5">
          <div class="flex items-center space-x-1.5">
            <span class="text-[11px] font-mono font-bold bg-slate-900 text-white px-1.5 py-0.5 rounded">
              Line #${pair.line_id}
            </span>
            <span class="text-[10px] font-medium text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded uppercase">
              ${pair.type}
            </span>
            ${pair.article_number ? `<span class="text-[11px] font-semibold text-slate-800 bg-slate-100 px-1.5 py-0.5 rounded">Art. ${pair.article_number}</span>` : ''}
            <button class="btn-jump-page text-[11px] font-medium text-slate-500 hover:text-slate-900 underline ml-1" data-page="${pair.page_number}">
              Page ${pair.page_number}
            </button>
          </div>
          
          <div class="flex items-center space-x-1.5">
            ${statusBadge}
            <button class="btn-verify px-2 py-0.5 text-xs font-semibold rounded bg-slate-900 hover:bg-slate-800 text-white transition" data-id="${pair.id}">
              Approve
            </button>
            <button class="btn-flag px-2 py-0.5 text-xs font-medium rounded border border-slate-300 hover:bg-slate-50 text-slate-700 transition" data-id="${pair.id}">
              Flag
            </button>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="text-[10px] font-bold text-slate-600 uppercase tracking-wider">
                Amharic (አማርኛ)
              </span>
            </div>
            <textarea 
              class="amh-input w-full p-2 text-xs font-amharic leading-relaxed rounded border border-slate-200 focus:border-slate-800 focus:ring-0 bg-slate-50/60 hover:bg-white transition resize-y"
              rows="3" 
              data-id="${pair.id}"
            >${pair.amharic}</textarea>
          </div>

          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="text-[10px] font-bold text-slate-600 uppercase tracking-wider">
                English
              </span>
            </div>
            <textarea 
              class="eng-input w-full p-2 text-xs leading-relaxed rounded border border-slate-200 focus:border-slate-800 focus:ring-0 bg-slate-50/60 hover:bg-white transition resize-y"
              rows="3" 
              data-id="${pair.id}"
            >${pair.english}</textarea>
          </div>
        </div>
      </div>
    `;
  }

  bindPairCardEvents() {
    if (!this.currentSession) return;

    document.querySelectorAll('.pair-card').forEach(card => {
      card.addEventListener('click', (e) => {
        const id = card.getAttribute('data-pair-id');
        const pageAttr = card.getAttribute('data-page');
        const pageNum = pageAttr ? parseInt(pageAttr, 10) : 1;
        this.setActiveCard(id, pageNum);
      });
    });

    document.querySelectorAll('.btn-jump-page').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const pageAttr = e.currentTarget.getAttribute('data-page');
        const pageNum = pageAttr ? parseInt(pageAttr, 10) : 1;
        this.activePageNum = pageNum;
        this.updatePdfPreview();
      });
    });

    document.querySelectorAll('.amh-input, .eng-input').forEach(textarea => {
      textarea.addEventListener('focus', (e) => {
        const card = e.target.closest('.pair-card');
        if (card) {
          const id = card.getAttribute('data-pair-id');
          const pageAttr = card.getAttribute('data-page');
          const pageNum = pageAttr ? parseInt(pageAttr, 10) : 1;
          this.setActiveCard(id, pageNum);
        }
      });
    });

    document.querySelectorAll('.btn-verify').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const id = e.currentTarget.getAttribute('data-id');
        if (!id) return;
        await this.handlePairUpdate(id, { status: 'verified' });
      });
    });

    document.querySelectorAll('.btn-flag').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const id = e.currentTarget.getAttribute('data-id');
        if (!id) return;
        await this.handlePairUpdate(id, { status: 'flagged', notes: 'Flagged by reviewer' });
      });
    });

    document.querySelectorAll('.amh-input').forEach(textarea => {
      textarea.addEventListener('change', async (e) => {
        const id = e.target.getAttribute('data-id');
        if (!id) return;
        await this.handlePairUpdate(id, { amharic: e.target.value });
      });
    });

    document.querySelectorAll('.eng-input').forEach(textarea => {
      textarea.addEventListener('change', async (e) => {
        const id = e.target.getAttribute('data-id');
        if (!id) return;
        await this.handlePairUpdate(id, { english: e.target.value });
      });
    });
  }

  setActiveCard(pairId, pageNum) {
    this.activePairId = pairId;
    document.querySelectorAll('.pair-card').forEach(c => {
      if (c.getAttribute('data-pair-id') === pairId) {
        c.classList.add('border-slate-800', 'ring-2', 'ring-slate-900/10');
      } else {
        c.classList.remove('border-slate-800', 'ring-2', 'ring-slate-900/10');
      }
    });

    if (pageNum && pageNum !== this.activePageNum) {
      this.activePageNum = pageNum;
      this.updatePdfPreview();
    }
  }

  async handlePairUpdate(pairId, updates) {
    if (!this.currentSession) return;
    try {
      const res = await api.updatePair(this.currentSession.session_id, pairId, updates);
      const idx = this.currentSession.pairs.findIndex(p => p.id === pairId);
      if (idx !== -1) {
        this.currentSession.pairs[idx] = res.pair;
      }
      this.currentSession.audit = res.audit;
      this.renderMetrics();
      this.renderPairs();
      this.showToast('Saved', 'info');
    } catch (e) {
      this.showToast('Save failed: ' + e.message, 'error');
    }
  }

  async batchApproveAll() {
    if (!this.currentSession) return;
    try {
      const ids = this.currentSession.pairs.map(p => p.id);
      const res = await api.batchApprove(this.currentSession.session_id, ids);
      this.currentSession.pairs.forEach(p => p.status = 'verified');
      this.currentSession.audit = res.audit;
      this.renderMetrics();
      this.renderPairs();
      this.showToast(`Batch approved ${res.verified_count} blocks`, 'success');
    } catch (e) {
      this.showToast('Batch approve failed: ' + e.message, 'error');
    }
  }

  updatePdfPreview() {
    if (!this.currentSession) return;
    const img = document.getElementById('pdf-preview-img');
    const label = document.getElementById('pdf-page-label');

    const maxPages = this.currentSession.total_pages || 10;
    this.activePageNum = Math.max(1, Math.min(this.activePageNum, maxPages));

    if (img) {
      img.src = api.getPageImageUrl(this.currentSession.session_id, this.activePageNum, this.activePdfLang);
    }
    if (label) {
      label.textContent = `Page ${this.activePageNum} / ${maxPages}`;
    }
  }

  setLoading(loading) {
    const spinner = document.getElementById('loading-overlay');
    if (spinner) {
      spinner.classList.toggle('hidden', !loading);
    }
  }

  showToast(msg, type = 'info') {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.className = `fixed bottom-4 right-4 px-3 py-2 rounded text-xs font-medium shadow-md z-50 transition-all duration-200 ${
      type === 'error' ? 'bg-rose-900 text-white' : 'bg-slate-900 text-white'
    }`;
    toast.classList.remove('translate-y-16', 'opacity-0');
    setTimeout(() => {
      toast.classList.add('translate-y-16', 'opacity-0');
    }, 2500);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new LegalHILApp();
});

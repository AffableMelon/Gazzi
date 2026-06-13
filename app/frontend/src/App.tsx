import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { api } from './api';
import { 
  HILSession, AlignedPair, ExampleDocument, 
  InputType, HILStatus 
} from './types';

export default function App() {
  const [examples, setExamples] = useState<ExampleDocument[]>([]);
  const [selectedMode, setSelectedMode] = useState<InputType>('gazette_bilingual');
  const [selectedDocId, setSelectedDocId] = useState<string>('');
  const [pageStart, setPageStart] = useState<number>(1);
  const [pageEnd, setPageEnd] = useState<number>(5);

  const [session, setSession] = useState<HILSession | null>(null);
  const [activePairId, setActivePairId] = useState<string | null>(null);
  const [activePage, setActivePage] = useState<number>(1);
  const [pdfLang, setPdfLang] = useState<'amh' | 'eng'>('amh');
  const [filter, setFilter] = useState<'all' | 'flagged' | 'verified' | 'pending'>('all');
  const [search, setSearch] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [toast, setToast] = useState<string | null>(null);

  // Upload Modal State
  const [isUploadOpen, setIsUploadOpen] = useState<boolean>(false);
  const [uploadFile1, setUploadFile1] = useState<File | null>(null);
  const [uploadFile2, setUploadFile2] = useState<File | null>(null);
  const [uploadJsonText, setUploadJsonText] = useState<string>('');
  const [uploadJsonFile, setUploadJsonFile] = useState<File | null>(null);
  const [jsonInputTab, setJsonInputTab] = useState<'text' | 'file'>('text');
  const [copiedPrompt, setCopiedPrompt] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);

  const fileInputRef1 = useRef<HTMLInputElement>(null);
  const fileInputRef2 = useRef<HTMLInputElement>(null);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  }, []);

  // Filtered documents for current mode
  const currentDocs = useMemo(() => {
    return examples.filter(e => e.input_type === selectedMode);
  }, [examples, selectedMode]);

  // Load available documents safely on mount
  useEffect(() => {
    const init = async () => {
      try {
        const docs = await api.getExamples();
        setExamples(docs);
        const first = docs.find(d => d.input_type === 'gazette_bilingual');
        if (first) {
          setSelectedDocId(first.id);
          await loadDocument(first.id, 'gazette_bilingual', 1, 5);
        } else {
          setSelectedDocId('');
          setSession(null);
        }
      } catch (err) {
        setExamples([]);
        setSelectedDocId('');
        setSession(null);
      }
    };
    init();
  }, []);

  // Execute or load document
  const loadDocument = async (docId: string, mode: InputType, pStart = pageStart, pEnd = pageEnd) => {
    if (!docId) return;
    setLoading(true);
    try {
      const res = await api.runPipeline({
        input_type: mode,
        example_id: docId,
        page_start: pStart,
        page_end: pEnd
      });
      setSession(res);
      const initialPage = res.pages_processed[0] || 1;
      setActivePage(initialPage);
      if (res.pairs.length > 0) {
        setActivePairId(res.pairs[0].id);
      }
      showToast(`Loaded "${res.document_id}"`);
    } catch (err: any) {
      showToast(err.message || 'Failed to load document');
      setSession(null);
    } finally {
      setLoading(false);
    }
  };

  // Mode change
  const handleModeChange = async (newMode: InputType) => {
    if (newMode === selectedMode) return;
    setSelectedMode(newMode);
    const docs = examples.filter(e => e.input_type === newMode);
    if (docs.length > 0) {
      setSelectedDocId(docs[0].id);
      await loadDocument(docs[0].id, newMode);
    } else {
      setSelectedDocId('');
      setSession(null);
    }
  };

  // Dropdown document change
  const handleDocChange = async (docId: string) => {
    if (!docId) return;
    setSelectedDocId(docId);
    await loadDocument(docId, selectedMode);
  };

  // Copy Gemini extraction prompt for current mode
  const handleCopyPrompt = async () => {
    try {
      const prompts = await api.getPrompts();
      const p = selectedMode === 'gazette_bilingual' ? prompts.gazette_prompt : prompts.codebook_prompt;
      await navigator.clipboard.writeText(p);
      setCopiedPrompt(true);
      showToast('Gemini prompt copied to clipboard!');
      setTimeout(() => setCopiedPrompt(false), 2200);
    } catch (e) {
      showToast('Failed to copy prompt');
    }
  };

  // Handle PDF Upload Submission with optional Gemini Chat JSON
  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile1) {
      showToast('Please select a PDF file');
      return;
    }
    if (selectedMode === 'codebook_dual' && !uploadFile2) {
      showToast('Please select both Amharic and English PDF files for dual mode');
      return;
    }

    setIsUploading(true);
    setLoading(true);
    try {
      const uploadRes = await api.uploadPdf(
        uploadFile1, 
        uploadFile2 || undefined,
        uploadJsonText.trim() || undefined,
        uploadJsonFile || undefined
      );
      
      const res = await api.runPipeline({
        input_type: selectedMode,
        uploaded_file_id: uploadRes.upload_id,
        uploaded_file_id_eng: uploadRes.file_path_eng ? uploadRes.upload_id : undefined,
        page_start: 1,
        page_end: 5
      });

      setSession(res);
      setActivePage(res.pages_processed[0] || 1);
      if (res.pairs.length > 0) setActivePairId(res.pairs[0].id);

      const freshDocs = await api.getExamples();
      setExamples(freshDocs);
      setSelectedDocId(res.session_id);

      setIsUploadOpen(false);
      setUploadFile1(null);
      setUploadFile2(null);
      setUploadJsonText('');
      setUploadJsonFile(null);
      showToast(uploadRes.has_json 
        ? `Loaded with Gemini Chat extraction: "${uploadFile1.name}"`
        : `Uploaded and processed "${uploadFile1.name}"`
      );
    } catch (err: any) {
      showToast('Upload error: ' + (err.message || 'Failed to ingest file'));
    } finally {
      setIsUploading(false);
      setLoading(false);
    }
  };

  // Touching/Clicking any pair card immediately jumps PDF to that page
  const handleSelectPair = (pair: AlignedPair) => {
    setActivePairId(pair.id);
    if (pair.page_number && pair.page_number !== activePage) {
      setActivePage(pair.page_number);
    }
  };

  // In-place text edits
  const handleTextChange = async (pairId: string, field: 'amharic' | 'english', value: string) => {
    if (!session) return;
    const updatedPairs = session.pairs.map(p => {
      if (p.id === pairId) {
        return { ...p, [field]: value, status: 'modified' as HILStatus };
      }
      return p;
    });

    setSession({ ...session, pairs: updatedPairs });

    try {
      const res = await api.updatePair(session.session_id, pairId, { [field]: value });
      setSession(prev => prev ? { ...prev, audit: res.audit } : prev);
    } catch (err) {
      console.error('Failed to auto-save change', err);
    }
  };

  // Approve single pair
  const handleApprove = async (pairId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!session) return;
    try {
      const res = await api.updatePair(session.session_id, pairId, { status: 'verified' });
      setSession(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          pairs: prev.pairs.map(p => p.id === pairId ? res.pair : p),
          audit: res.audit
        };
      });
      showToast('Verified');
    } catch (err: any) {
      showToast('Approval failed');
    }
  };

  // Flag single pair
  const handleFlag = async (pairId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!session) return;
    try {
      const res = await api.updatePair(session.session_id, pairId, { 
        status: 'flagged', 
        notes: 'Flagged for review' 
      });
      setSession(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          pairs: prev.pairs.map(p => p.id === pairId ? res.pair : p),
          audit: res.audit
        };
      });
      showToast('Flagged');
    } catch (err: any) {
      showToast('Flagging failed');
    }
  };

  // Batch approve all pairs
  const handleBatchApprove = async () => {
    if (!session || session.pairs.length === 0) return;
    try {
      const ids = session.pairs.map(p => p.id);
      const res = await api.batchApprove(session.session_id, ids);
      setSession(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          pairs: prev.pairs.map(p => ({ ...p, status: 'verified' as HILStatus })),
          audit: res.audit
        };
      });
      showToast(`Batch approved ${res.verified_count} blocks`);
    } catch (err: any) {
      showToast('Batch approval error');
    }
  };

  // Filtered pairs list
  const visiblePairs = useMemo(() => {
    if (!session) return [];
    let list = session.pairs;

    if (filter === 'flagged') list = list.filter(p => p.status === 'flagged');
    else if (filter === 'verified') list = list.filter(p => p.status === 'verified' || p.status === 'modified');
    else if (filter === 'pending') list = list.filter(p => p.status === 'pending');

    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(p => 
        p.amharic.toLowerCase().includes(q) ||
        p.english.toLowerCase().includes(q) ||
        (p.article_number && p.article_number.includes(q)) ||
        p.line_id.toString().includes(q)
      );
    }
    return list;
  }, [session, filter, search]);

  const totalPages = session?.total_pages || 0;

  return (
    <div className="min-h-screen flex flex-col bg-slate-100 text-slate-900 font-sans">
      
      {/* Top Header */}
      <header className="bg-slate-900 text-white border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-12 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="font-bold text-xs tracking-wider uppercase text-slate-100">Ethiopian Legal AI</span>
            <span className="text-slate-600">/</span>
            <span className="text-xs text-slate-400 font-medium">HIL Verification Workbench</span>
          </div>

          <div className="flex items-center space-x-2">
            <button 
              disabled={!session}
              onClick={() => session && window.open(api.getExportUrl(session.session_id, 'json'), '_blank')}
              className="px-2.5 py-1 text-xs font-medium rounded border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-200 transition disabled:opacity-40"
            >
              Export JSON
            </button>
            <button 
              disabled={!session}
              onClick={() => session && window.open(api.getExportUrl(session.session_id, 'csv'), '_blank')}
              className="px-2.5 py-1 text-xs font-medium rounded border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-200 transition disabled:opacity-40"
            >
              Export CSV
            </button>
          </div>
        </div>
      </header>

      {/* Control Toolbar */}
      <section className="bg-white border-b border-slate-200 py-2.5 px-4 sm:px-6 shadow-2xs">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          
          {/* Mode Switcher */}
          <div className="inline-flex rounded-lg border border-slate-200 bg-slate-100 p-0.5 text-xs font-medium">
            <button 
              onClick={() => handleModeChange('gazette_bilingual')}
              className={`px-3 py-1 rounded-md transition ${
                selectedMode === 'gazette_bilingual' 
                  ? 'bg-white text-slate-900 shadow-2xs font-semibold' 
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Bilingual Gazette (Single PDF)
            </button>
            <button 
              onClick={() => handleModeChange('codebook_dual')}
              className={`px-3 py-1 rounded-md transition ${
                selectedMode === 'codebook_dual' 
                  ? 'bg-white text-slate-900 shadow-2xs font-semibold' 
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Dual Codebooks (Amharic + English)
            </button>
          </div>

          {/* Document Picker, Upload Button & Controls */}
          <div className="flex items-center flex-wrap gap-2">
            
            {/* Upload Button */}
            <button 
              onClick={() => setIsUploadOpen(true)}
              className="px-3 py-1 rounded-md text-xs font-semibold border border-slate-300 bg-white hover:bg-slate-50 text-slate-800 transition shadow-2xs"
            >
              Upload PDF
            </button>

            {currentDocs.length > 0 ? (
              <div className="flex items-center space-x-1.5">
                <label className="text-xs text-slate-500 font-medium">Document:</label>
                <select 
                  value={selectedDocId}
                  onChange={(e) => handleDocChange(e.target.value)}
                  className="text-xs bg-slate-50 border border-slate-300 rounded-md px-2.5 py-1 text-slate-800 focus:outline-none focus:ring-1 focus:ring-slate-900 max-w-xs truncate"
                >
                  {currentDocs.map(d => (
                    <option key={d.id} value={d.id}>{d.title} ({d.pages} pages)</option>
                  ))}
                </select>
              </div>
            ) : (
              <span className="text-xs text-slate-400 italic px-2">No documents loaded yet</span>
            )}

            <div className="flex items-center space-x-1">
              <label className="text-xs text-slate-500">Pages:</label>
              <input 
                type="number" 
                value={pageStart}
                min={1}
                onChange={(e) => setPageStart(parseInt(e.target.value, 10) || 1)}
                className="w-10 text-xs text-center border border-slate-300 rounded py-1"
              />
              <span className="text-slate-400 text-xs">-</span>
              <input 
                type="number" 
                value={pageEnd}
                min={1}
                onChange={(e) => setPageEnd(parseInt(e.target.value, 10) || 1)}
                className="w-10 text-xs text-center border border-slate-300 rounded py-1"
              />
            </div>

            <button 
              disabled={!selectedDocId}
              onClick={() => selectedDocId && loadDocument(selectedDocId, selectedMode)}
              className="px-3 py-1 rounded-md text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-white transition shadow-2xs disabled:opacity-40"
            >
              Reload
            </button>
          </div>

        </div>
      </section>

      {/* Metrics Strip */}
      <section className="bg-white border-b border-slate-200 py-2 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          
          <div className="flex items-center space-x-6 text-xs">
            <div>
              <span className="text-slate-400 uppercase tracking-wider font-semibold text-[10px]">Total Blocks</span>
              <span className="font-bold text-slate-900 ml-1.5">{session?.audit.total_pairs || 0}</span>
            </div>
            <div>
              <span className="text-emerald-600 uppercase tracking-wider font-semibold text-[10px]">Verified</span>
              <span className="font-bold text-emerald-700 ml-1.5">
                {(session?.audit.verified_count || 0) + (session?.audit.modified_count || 0)}
              </span>
            </div>
            <div>
              <span className="text-amber-600 uppercase tracking-wider font-semibold text-[10px]">Needs Review</span>
              <span className="font-bold text-amber-700 ml-1.5">{session?.audit.flagged_count || 0}</span>
            </div>
            <div>
              <span className="text-slate-500 uppercase tracking-wider font-semibold text-[10px]">Match Score</span>
              <span className="font-bold text-slate-900 ml-1.5">
                {session && session.audit.total_pairs > 0 ? `${session.audit.match_accuracy_pct}%` : '0%'}
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 text-xs text-slate-500">
              <span>Progress:</span>
              <div className="w-24 bg-slate-200 rounded-full h-2 overflow-hidden">
                <div 
                  className="bg-emerald-600 h-2 rounded-full transition-all duration-300"
                  style={{ 
                    width: `${session?.audit.total_pairs 
                      ? Math.round((((session.audit.verified_count + session.audit.modified_count) / session.audit.total_pairs) * 100)) 
                      : 0}%` 
                  }}
                />
              </div>
              <span className="font-semibold text-slate-700">
                {session?.audit.total_pairs 
                  ? Math.round((((session.audit.verified_count + session.audit.modified_count) / session.audit.total_pairs) * 100)) 
                  : 0}%
              </span>
            </div>

            <button 
              disabled={!session || session.pairs.length === 0}
              onClick={handleBatchApprove}
              className="px-2.5 py-1 text-xs font-semibold rounded bg-emerald-600 hover:bg-emerald-700 text-white transition shadow-2xs disabled:opacity-40"
            >
              Batch Approve All
            </button>
          </div>

        </div>
      </section>

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-4">
        
        {/* Filter Toolbar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 gap-2">
          <div className="flex items-center space-x-1 text-xs">
            {(['all', 'flagged', 'verified', 'pending'] as const).map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-2.5 py-1 rounded font-medium transition ${
                  filter === f 
                    ? 'bg-slate-900 text-white font-semibold' 
                    : 'text-slate-600 hover:bg-slate-200'
                }`}
              >
                {f === 'all' ? 'All Blocks' : f === 'flagged' ? 'Needs Review' : f === 'verified' ? 'Verified' : 'Pending'}
              </button>
            ))}
          </div>

          <input 
            type="text" 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search text, article..." 
            className="text-xs px-2.5 py-1 rounded border border-slate-300 bg-white focus:outline-none focus:ring-1 focus:ring-slate-900 w-48"
          />
        </div>

        {/* 2-Column Split: Aligned Cards (Left) vs PDF Document Preview (Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
          
          {/* Left Column: Aligned Blocks */}
          <div className="lg:col-span-7 xl:col-span-7 space-y-3">
            {!session && !loading ? (
              <div className="text-center py-16 bg-white rounded-lg border border-dashed border-slate-300 p-6">
                <p className="text-sm font-semibold text-slate-800">No document currently loaded</p>
                <p className="text-xs text-slate-500 mt-1">Click below to upload a PDF and run the AI extraction.</p>
                <button 
                  onClick={() => setIsUploadOpen(true)}
                  className="mt-3 px-3 py-1.5 text-xs font-semibold rounded bg-slate-900 text-white hover:bg-slate-800 transition"
                >
                  Upload PDF Now
                </button>
              </div>
            ) : visiblePairs.length === 0 ? (
              <div className="text-center py-12 text-slate-400 bg-white rounded-lg border border-dashed border-slate-300 text-xs">
                No aligned blocks match the current filter.
              </div>
            ) : (
              visiblePairs.map(pair => {
                const isVerified = pair.status === 'verified';
                const isFlagged = pair.status === 'flagged';
                const isModified = pair.status === 'modified';
                const isActive = pair.id === activePairId;

                return (
                  <div
                    key={pair.id}
                    onClick={() => handleSelectPair(pair)}
                    className={`pair-card bg-white rounded-lg border p-3.5 cursor-pointer transition shadow-2xs hover:border-slate-400 ${
                      isActive 
                        ? 'border-slate-800 ring-2 ring-slate-900/10' 
                        : isFlagged 
                        ? 'border-amber-300' 
                        : 'border-slate-200'
                    }`}
                  >
                    {/* Discrepancy Notice */}
                    {pair.discrepancy_reason && (
                      <div className="mb-2 px-2.5 py-1 rounded bg-amber-50 border border-amber-200 text-[11px] text-amber-800 flex items-center justify-between">
                        <span><strong>Notice:</strong> {pair.discrepancy_reason}</span>
                        <span className="text-[10px] font-mono opacity-80">Score: {(pair.confidence * 100).toFixed(0)}%</span>
                      </div>
                    )}

                    {/* Card Header */}
                    <div className="flex items-center justify-between border-b border-slate-100 pb-2 mb-2.5">
                      <div className="flex items-center space-x-1.5">
                        <span className="text-[11px] font-mono font-bold bg-slate-900 text-white px-1.5 py-0.5 rounded">
                          Line #{pair.line_id}
                        </span>
                        <span className="text-[10px] font-medium text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded uppercase">
                          {pair.type}
                        </span>
                        {pair.article_number && (
                          <span className="text-[11px] font-semibold text-slate-800 bg-slate-100 px-1.5 py-0.5 rounded">
                            Art. {pair.article_number}
                          </span>
                        )}
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            setActivePage(pair.page_number);
                            setActivePairId(pair.id);
                          }}
                          className="text-[11px] font-medium text-slate-500 hover:text-slate-900 underline ml-1"
                        >
                          Page {pair.page_number}
                        </button>
                      </div>

                      <div className="flex items-center space-x-1.5">
                        <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold border ${
                          isVerified ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                          isModified ? 'bg-sky-50 text-sky-700 border-sky-200' :
                          isFlagged ? 'bg-amber-50 text-amber-700 border-amber-300' :
                          'bg-slate-100 text-slate-600 border-slate-200'
                        }`}>
                          {isVerified ? 'Verified' : isModified ? 'Edited' : isFlagged ? 'Review Needed' : 'Pending'}
                        </span>

                        <button 
                          onClick={(e) => handleApprove(pair.id, e)}
                          className="px-2 py-0.5 text-xs font-semibold rounded bg-slate-900 hover:bg-slate-800 text-white transition"
                        >
                          Approve
                        </button>
                        <button 
                          onClick={(e) => handleFlag(pair.id, e)}
                          className="px-2 py-0.5 text-xs font-medium rounded border border-slate-300 hover:bg-slate-50 text-slate-700 transition"
                        >
                          Flag
                        </button>
                      </div>
                    </div>

                    {/* Dual Editable Columns */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[10px] font-bold text-slate-600 uppercase tracking-wider">
                            Amharic (አማርኛ)
                          </span>
                        </div>
                        <textarea 
                          value={pair.amharic}
                          onFocus={() => handleSelectPair(pair)}
                          onChange={(e) => handleTextChange(pair.id, 'amharic', e.target.value)}
                          className="w-full p-2 text-xs font-amharic leading-relaxed rounded border border-slate-200 focus:border-slate-800 focus:ring-0 bg-slate-50/60 hover:bg-white transition resize-y"
                          rows={3}
                        />
                      </div>

                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[10px] font-bold text-slate-600 uppercase tracking-wider">
                            English
                          </span>
                        </div>
                        <textarea 
                          value={pair.english}
                          onFocus={() => handleSelectPair(pair)}
                          onChange={(e) => handleTextChange(pair.id, 'english', e.target.value)}
                          className="w-full p-2 text-xs leading-relaxed rounded border border-slate-200 focus:border-slate-800 focus:ring-0 bg-slate-50/60 hover:bg-white transition resize-y"
                          rows={3}
                        />
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Right Column: Synchronized Scanned PDF Preview */}
          <div className="lg:col-span-5 xl:col-span-5 sticky top-4">
            <div className="bg-white rounded-lg border border-slate-300 shadow-sm overflow-hidden">
              
              {/* PDF Viewer Header */}
              <div className="px-3 py-2 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-bold text-slate-800">Scanned Document</span>
                  
                  {/* Type 2 Language Toggle */}
                  {selectedMode === 'codebook_dual' && (
                    <div className="inline-flex items-center text-[11px] border border-slate-300 rounded bg-white overflow-hidden">
                      <button 
                        onClick={() => setPdfLang('amh')}
                        className={`px-2 py-0.5 font-semibold transition ${
                          pdfLang === 'amh' ? 'bg-slate-800 text-white' : 'text-slate-600 hover:bg-slate-100'
                        }`}
                      >
                        Amharic
                      </button>
                      <button 
                        onClick={() => setPdfLang('eng')}
                        className={`px-2 py-0.5 font-semibold transition ${
                          pdfLang === 'eng' ? 'bg-slate-800 text-white' : 'text-slate-600 hover:bg-slate-100'
                        }`}
                      >
                        English
                      </button>
                    </div>
                  )}
                </div>

                {/* Page Navigation */}
                <div className="flex items-center space-x-1">
                  <button 
                    disabled={!session || activePage <= 1}
                    onClick={() => setActivePage(p => Math.max(1, p - 1))}
                    className="px-2 py-0.5 text-xs font-semibold rounded border border-slate-300 bg-white hover:bg-slate-100 disabled:opacity-40 text-slate-700"
                  >
                    Prev
                  </button>
                  <span className="text-xs font-mono font-medium text-slate-600 px-1">
                    {session ? `Page ${activePage} / ${totalPages}` : '- / -'}
                  </span>
                  <button 
                    disabled={!session || activePage >= totalPages}
                    onClick={() => setActivePage(p => Math.min(totalPages, p + 1))}
                    className="px-2 py-0.5 text-xs font-semibold rounded border border-slate-300 bg-white hover:bg-slate-100 disabled:opacity-40 text-slate-700"
                  >
                    Next
                  </button>
                </div>
              </div>
              
              {/* Image Canvas */}
              <div className="p-2 bg-slate-200/50 max-h-[78vh] overflow-y-auto flex items-center justify-center min-h-[420px]">
                {session ? (
                  <img 
                    key={`${session.session_id}_${activePage}_${pdfLang}`}
                    src={api.getPageImageUrl(session.session_id, activePage, pdfLang)}
                    alt={`Page ${activePage}`}
                    className="max-w-full rounded border border-slate-300 shadow-xs object-contain bg-white"
                  />
                ) : (
                  <div className="text-center p-6 text-slate-400 text-xs">
                    <p className="font-medium text-slate-500 mb-1">No Scanned Document</p>
                    <p>Upload a PDF to view original scanned pages here.</p>
                  </div>
                )}
              </div>

              <div className="px-3 py-1.5 bg-slate-50 border-t border-slate-200 text-center text-[11px] text-slate-500">
                {session ? 'Clicking any section on the left jumps to its page.' : 'No active document.'}
              </div>
            </div>
          </div>

        </div>
      </main>

      {/* PDF Upload Modal */}
      {isUploadOpen && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-lg w-full border border-slate-200 p-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <h3 className="text-sm font-bold text-slate-900">
                Upload {selectedMode === 'gazette_bilingual' ? 'Bilingual Gazette' : 'Dual Codebooks'}
              </h3>
              <button 
                onClick={() => setIsUploadOpen(false)}
                className="text-slate-400 hover:text-slate-700 text-xs font-bold"
              >
                Close
              </button>
            </div>

            <form onSubmit={handleUploadSubmit} className="mt-4 space-y-4 text-xs">
              {selectedMode === 'gazette_bilingual' ? (
                <div>
                  <label className="block font-medium text-slate-700 mb-1">
                    Bilingual Proclamation / Gazette PDF:
                  </label>
                  <input 
                    type="file" 
                    accept=".pdf"
                    ref={fileInputRef1}
                    onChange={(e) => setUploadFile1(e.target.files?.[0] || null)}
                    className="w-full text-xs text-slate-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-slate-900 file:text-white hover:file:bg-slate-800"
                  />
                  <p className="text-[11px] text-slate-400 mt-1">Single PDF with dual columns (Amharic on left, English on right).</p>
                </div>
              ) : (
                <>
                  <div>
                    <label className="block font-medium text-slate-700 mb-1">
                      1. Amharic Codebook PDF:
                    </label>
                    <input 
                      type="file" 
                      accept=".pdf"
                      ref={fileInputRef1}
                      onChange={(e) => setUploadFile1(e.target.files?.[0] || null)}
                      className="w-full text-xs text-slate-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-slate-900 file:text-white hover:file:bg-slate-800"
                    />
                  </div>

                  <div>
                    <label className="block font-medium text-slate-700 mb-1">
                      2. English Codebook PDF:
                    </label>
                    <input 
                      type="file" 
                      accept=".pdf"
                      ref={fileInputRef2}
                      onChange={(e) => setUploadFile2(e.target.files?.[0] || null)}
                      className="w-full text-xs text-slate-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-slate-900 file:text-white hover:file:bg-slate-800"
                    />
                  </div>
                </>
              )}

              {/* Gemini Chat Extracted JSON Integration */}
              <div className="pt-3 border-t border-slate-200">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold text-slate-800 text-[11px] uppercase tracking-wider">
                    Gemini Chat JSON (Optional / Fast-Track)
                  </span>
                  <button
                    type="button"
                    onClick={handleCopyPrompt}
                    className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold border border-slate-300 bg-slate-50 hover:bg-slate-100 text-slate-800 transition shadow-2xs"
                  >
                    {copiedPrompt ? 'Copied Prompt!' : 'Copy Gemini Prompt'}
                  </button>
                </div>
                <p className="text-[11px] text-slate-500 mb-2">
                  Attach your PDF to Gemini Web Chat with the prompt, then paste Gemini's returned JSON below to verify it immediately in the workbench.
                </p>

                <div className="flex items-center space-x-2 mb-2">
                  <button
                    type="button"
                    onClick={() => setJsonInputTab('text')}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                      jsonInputTab === 'text' ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    Paste JSON Text
                  </button>
                  <button
                    type="button"
                    onClick={() => setJsonInputTab('file')}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                      jsonInputTab === 'file' ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    Upload .json File
                  </button>
                </div>

                {jsonInputTab === 'text' ? (
                  <textarea
                    value={uploadJsonText}
                    onChange={(e) => setUploadJsonText(e.target.value)}
                    rows={4}
                    placeholder="Paste JSON output from Gemini chat here... (markdown ```json code blocks are automatically stripped)"
                    className="w-full text-xs font-mono p-2 border border-slate-300 rounded focus:border-slate-900 focus:outline-none bg-slate-50/50"
                  />
                ) : (
                  <input
                    type="file"
                    accept=".json"
                    onChange={(e) => setUploadJsonFile(e.target.files?.[0] || null)}
                    className="w-full text-xs text-slate-500 file:mr-3 file:py-1 file:px-2.5 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-slate-100 file:text-slate-800 hover:file:bg-slate-200"
                  />
                )}
              </div>

              <div className="pt-2 flex items-center justify-end space-x-2 border-t border-slate-100">
                <button 
                  type="button" 
                  onClick={() => setIsUploadOpen(false)}
                  className="px-3 py-1.5 text-xs font-medium rounded border border-slate-300 text-slate-600 hover:bg-slate-50"
                >
                  Cancel
                </button>
                <button 
                  type="submit" 
                  disabled={isUploading}
                  className="px-3.5 py-1.5 text-xs font-semibold rounded bg-slate-900 hover:bg-slate-800 text-white transition disabled:opacity-50"
                >
                  {isUploading ? 'Ingesting PDF...' : 'Ingest & Process'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Loading Overlay */}
      {loading && (
        <div className="fixed inset-0 bg-slate-900/30 backdrop-blur-xs flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-5 max-w-xs w-full shadow-lg text-center border border-slate-200">
            <div className="w-8 h-8 border-2 border-slate-900 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            <p className="text-xs font-semibold text-slate-800">Processing Document...</p>
          </div>
        </div>
      )}

      {/* Toast Notification */}
      {toast && (
        <div className="fixed bottom-4 right-4 px-3 py-2 rounded text-xs font-medium shadow-md z-50 bg-slate-900 text-white transition-all">
          {toast}
        </div>
      )}

    </div>
  );
}

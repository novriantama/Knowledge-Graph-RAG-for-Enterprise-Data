import React, { useState, useEffect } from 'react';
import { 
  Network, 
  Search, 
  FileText, 
  BarChart3, 
  Send, 
  CheckCircle2, 
  AlertCircle, 
  Loader2, 
  Sparkles,
  Database,
  ShieldCheck,
  Zap
} from 'lucide-react';

interface QueryResult {
  question: string;
  answer: string;
  citations: string[];
  route_used: string;
  retrieved_chunk_ids: string[];
}

interface IngestResult {
  document_id: string;
  processed_chunks_count: number;
  status: string;
}

const API_BASE_URL = 'http://localhost:8000/api/v1';

const PRESET_QUESTIONS = [
  { label: '3-Hop Transitive', text: 'Which packages affect EU CRA compliance for Acme EU GmbH?' },
  { label: '2-Hop Relational', text: 'How is API Gateway Service connected to Stripe payment processing?' },
  { label: '1-Hop Single Fact', text: 'Where is Acme EU GmbH located?' },
  { label: 'Aggregation', text: 'Count how many microservices depend on Redis Cluster.' },
  { label: 'Out of Scope', text: 'What is the quarterly revenue of Tesla Motors in 2025?' },
];

export function App() {
  const [activeTab, setActiveTab] = useState<'query' | 'ingest' | 'benchmark'>('query');
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);

  // Query State
  const [queryInput, setQueryInput] = useState('');
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);

  // Ingest State
  const [docId, setDocId] = useState('');
  const [docContent, setDocContent] = useState('');
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);

  // Benchmark State
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkTableMd, setBenchmarkTableMd] = useState<string | null>(null);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);

  // Health Check
  useEffect(() => {
    fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'ping' }),
    })
      .then(() => setApiConnected(true))
      .catch(() => setApiConnected(false));
  }, []);

  const handleRunQuery = async (questionText?: string) => {
    const q = questionText || queryInput;
    if (!q.trim()) return;

    setQueryLoading(true);
    setQueryError(null);
    setQueryResult(null);

    try {
      const res = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data: QueryResult = await res.json();
      setQueryResult(data);
      setApiConnected(true);
    } catch (err: any) {
      setQueryError(err.message || 'Failed to execute query');
      setApiConnected(false);
    } finally {
      setQueryLoading(false);
    }
  };

  const handleIngestDocument = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!docId.trim() || !docContent.trim()) return;

    setIngestLoading(true);
    setIngestError(null);
    setIngestResult(null);

    const chunks = docContent.split('\n\n').filter(c => c.trim().length > 0);

    try {
      const res = await fetch(`${API_BASE_URL}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: docId.trim(),
          text_chunks: chunks,
          section_paths: chunks.map((_, i) => `/section_${i + 1}`)
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data: IngestResult = await res.json();
      setIngestResult(data);
      setDocId('');
      setDocContent('');
    } catch (err: any) {
      setIngestError(err.message || 'Failed to ingest document');
    } finally {
      setIngestLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    setBenchmarkLoading(true);
    setBenchmarkError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ questions_file_path: 'data/benchmark_questions.json' }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      setBenchmarkTableMd(data.summary_markdown);
    } catch (err: any) {
      setBenchmarkError(err.message || 'Failed to run benchmark');
    } finally {
      setBenchmarkLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-title-section">
          <div className="header-icon">
            <Network size={22} />
          </div>
          <div className="header-title">
            <h1>Enterprise Knowledge Graph RAG</h1>
            <p>Hybrid Neo4j Graph + pgvector HNSW Engine</p>
          </div>
        </div>

        <div className="api-status-badge">
          <div className={`status-dot ${apiConnected === false ? 'offline' : ''}`} />
          <span>API {apiConnected ? 'Online (http://localhost:8000)' : 'Connecting...'}</span>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="tab-navigation">
        <button 
          className={`tab-button ${activeTab === 'query' ? 'active' : ''}`}
          onClick={() => setActiveTab('query')}
        >
          <Search size={17} />
          <span>Ask Pipeline Query</span>
        </button>

        <button 
          className={`tab-button ${activeTab === 'ingest' ? 'active' : ''}`}
          onClick={() => setActiveTab('ingest')}
        >
          <FileText size={17} />
          <span>Ingest Document</span>
        </button>

        <button 
          className={`tab-button ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          <BarChart3 size={17} />
          <span>50-Item Benchmark</span>
        </button>
      </nav>

      {/* TAB 1: Query Pipeline */}
      {activeTab === 'query' && (
        <div>
          <div className="glass-panel">
            <div className="input-group">
              <label className="input-label">Enterprise Question</label>
              <textarea 
                className="textarea-input"
                placeholder="Ask a multi-hop, relational, or single-fact question..."
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    handleRunQuery();
                  }
                }}
              />
            </div>

            <button 
              className="btn-primary" 
              onClick={() => handleRunQuery()}
              disabled={queryLoading || !queryInput.trim()}
            >
              {queryLoading ? <Loader2 className="animate-spin" size={18} /> : <Send size={18} />}
              <span>{queryLoading ? 'Routing & Executing...' : 'Run Query'}</span>
            </button>

            {/* Presets */}
            <div className="presets-container">
              <div className="presets-label">Sample Questions:</div>
              <div className="presets-grid">
                {PRESET_QUESTIONS.map((item, idx) => (
                  <button
                    key={idx}
                    className="preset-chip"
                    onClick={() => {
                      setQueryInput(item.text);
                      handleRunQuery(item.text);
                    }}
                  >
                    <strong>{item.label}:</strong> "{item.text}"
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Error Notice */}
          {queryError && (
            <div className="glass-panel" style={{ borderColor: 'rgba(244, 63, 94, 0.4)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#f43f5e' }}>
                <AlertCircle size={20} />
                <span>{queryError}</span>
              </div>
            </div>
          )}

          {/* Grounded Result Display */}
          {queryResult && (
            <div className="glass-panel" style={{ borderLeft: '4px solid #00f2fe' }}>
              <div className="result-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <Sparkles size={20} style={{ color: '#00f2fe' }} />
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Grounded Answer Output</h3>
                </div>
                
                <span className={`route-badge ${queryResult.route_used.toLowerCase()}`}>
                  {queryResult.route_used} ROUTE
                </span>
              </div>

              <div className="answer-body">
                {queryResult.answer}
              </div>

              {/* Citations */}
              {queryResult.citations && queryResult.citations.length > 0 && (
                <div className="citations-section">
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <ShieldCheck size={16} style={{ color: '#10b981' }} />
                    <span>Validated Citations ({queryResult.citations.length}):</span>
                  </div>
                  <div className="citations-list">
                    {queryResult.citations.map((c, i) => (
                      <span key={i} className="citation-chip">{c}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Retrieved Chunks */}
              {queryResult.retrieved_chunk_ids && queryResult.retrieved_chunk_ids.length > 0 && (
                <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                  <div style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: '0.5rem' }}>
                    Shared Chunk IDs Retracted from Neo4j & pgvector:
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                    {queryResult.retrieved_chunk_ids.map((id, i) => (
                      <span key={i} style={{ fontSize: '0.75rem', color: '#94a3b8', background: 'rgba(255,255,255,0.04)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                        {id}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* TAB 2: Ingest Document */}
      {activeTab === 'ingest' && (
        <div className="glass-panel">
          <form onSubmit={handleIngestDocument}>
            <div className="input-group">
              <label className="input-label">Document ID</label>
              <input 
                type="text"
                className="text-input"
                placeholder="e.g. 05_disaster_recovery_policy"
                value={docId}
                onChange={(e) => setDocId(e.target.value)}
                required
              />
            </div>

            <div className="input-group">
              <label className="input-label">Document Content (Paragraphs separated by double line breaks)</label>
              <textarea 
                className="textarea-input"
                style={{ minHeight: '180px' }}
                placeholder="Paste raw enterprise document text here..."
                value={docContent}
                onChange={(e) => setDocContent(e.target.value)}
                required
              />
            </div>

            <button 
              type="submit" 
              className="btn-primary"
              disabled={ingestLoading || !docId.trim() || !docContent.trim()}
            >
              {ingestLoading ? <Loader2 className="animate-spin" size={18} /> : <Database size={18} />}
              <span>{ingestLoading ? 'Ingesting into Neo4j & pgvector...' : 'Ingest Document'}</span>
            </button>
          </form>

          {ingestResult && (
            <div className="glass-panel" style={{ marginTop: '1.5rem', borderColor: 'rgba(16, 185, 129, 0.4)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#10b981' }}>
                <CheckCircle2 size={20} />
                <div>
                  <strong>Document Ingested Successfully!</strong>
                  <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '0.25rem' }}>
                    Document '{ingestResult.document_id}' created {ingestResult.processed_chunks_count} chunks across Neo4j nodes and pgvector HNSW embeddings.
                  </p>
                </div>
              </div>
            </div>
          )}

          {ingestError && (
            <div className="glass-panel" style={{ marginTop: '1.5rem', borderColor: 'rgba(244, 63, 94, 0.4)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#f43f5e' }}>
                <AlertCircle size={20} />
                <span>{ingestError}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: Benchmark Suite */}
      {activeTab === 'benchmark' && (
        <div className="glass-panel">
          <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '0.5rem' }}>
              50-Item Stratified Portfolio Benchmark
            </h3>
            <p style={{ fontSize: '0.9rem', color: '#94a3b8' }}>
              Run accuracy delta evaluation comparing Plain Vector RAG vs Hybrid KG-RAG across 1-hop, 2-hop, 3-hop, aggregation, and out-of-scope refusals.
            </p>
          </div>

          <button 
            className="btn-primary" 
            onClick={handleRunBenchmark}
            disabled={benchmarkLoading}
          >
            {benchmarkLoading ? <Loader2 className="animate-spin" size={18} /> : <Zap size={18} />}
            <span>{benchmarkLoading ? 'Running 50-Item Benchmark Suite...' : 'Execute Comparative Benchmark'}</span>
          </button>

          {benchmarkError && (
            <div className="glass-panel" style={{ marginTop: '1.5rem', borderColor: 'rgba(244, 63, 94, 0.4)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#f43f5e' }}>
                <AlertCircle size={20} />
                <span>{benchmarkError}</span>
              </div>
            </div>
          )}

          {benchmarkTableMd ? (
            <div style={{ marginTop: '1.5rem', overflowX: 'auto' }}>
              <div className="input-label" style={{ marginBottom: '0.75rem' }}>Latest Benchmark Summary</div>
              <pre style={{ 
                background: 'rgba(11, 15, 25, 0.8)', 
                padding: '1.25rem', 
                borderRadius: '8px', 
                fontSize: '0.85rem', 
                color: '#00f2fe',
                border: '1px solid var(--border-color)',
                lineHeight: '1.6'
              }}>
                {benchmarkTableMd}
              </pre>
            </div>
          ) : (
            <div style={{ marginTop: '1.5rem' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Query Complexity</th>
                    <th>Plain Vector RAG Acc</th>
                    <th>Hybrid KG-RAG Acc</th>
                    <th>Accuracy Delta</th>
                    <th>Vector Latency</th>
                    <th>KG Latency</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>1-Hop (Single Fact)</strong></td>
                    <td>90.0%</td>
                    <td>80.0%</td>
                    <td className="delta-negative">-10.0%</td>
                    <td>3.62s</td>
                    <td>8.40s</td>
                  </tr>
                  <tr>
                    <td><strong>2-Hop (Relational)</strong></td>
                    <td>90.0%</td>
                    <td>70.0%</td>
                    <td className="delta-negative">-20.0%</td>
                    <td>5.05s</td>
                    <td>9.84s</td>
                  </tr>
                  <tr>
                    <td><strong>3-Hop (Transitive Chain)</strong></td>
                    <td>90.0%</td>
                    <td>100.0%</td>
                    <td className="delta-positive">+10.0%</td>
                    <td>8.77s</td>
                    <td>15.04s</td>
                  </tr>
                  <tr>
                    <td><strong>Aggregation / Grouping</strong></td>
                    <td>60.0%</td>
                    <td>60.0%</td>
                    <td className="delta-neutral">+0.0%</td>
                    <td>6.59s</td>
                    <td>11.52s</td>
                  </tr>
                  <tr>
                    <td><strong>Out of Scope (Refusal)</strong></td>
                    <td>60.0%</td>
                    <td>70.0%</td>
                    <td className="delta-positive">+10.0%</td>
                    <td>4.26s</td>
                    <td>8.02s</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;

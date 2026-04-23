'use client';
import { useState } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { ToastProvider, useToast } from '@/components/ui/Toast';
import { textBiasService, TextBiasResponse } from '@/services';
import { FileText, Play, ShieldAlert } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';

function TextBiasContent() {
  const [text, setText] = useState('The doctor told the nurse that she was wrong.');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TextBiasResponse | null>(null);
  const { toast } = useToast();

  const handleAnalyze = async () => {
    if (!text.trim()) {
      toast('warning', 'Empty Input', 'Please enter text to analyze.');
      return;
    }
    setLoading(true);
    try {
      const res = await textBiasService.analyzeText({ text });
      setResult(res);
      toast('success', 'Analysis Complete');
    } catch (e: any) {
      toast('error', 'Analysis Failed', e.message);
    } finally {
      setLoading(false);
    }
  };

  const renderHighlightedText = () => {
    if (!result) return null;
    let currentIdx = 0;
    const parts = [];

    // Sort tokens by start_idx to safely interleave
    const sortedTokens = [...result.biased_tokens].sort((a, b) => a.start_idx - b.start_idx);

    for (const token of sortedTokens) {
      // Add text before token
      if (token.start_idx > currentIdx) {
        parts.push(text.substring(currentIdx, token.start_idx));
      }
      // Add highlighted token
      parts.push(
        <span key={token.start_idx} style={{ 
          background: 'rgba(239,68,68,0.2)', 
          borderBottom: '2px solid #ef4444', 
          padding: '0 2px', 
          cursor: 'pointer',
          position: 'relative'
        }} title={`Bias Type: ${token.bias_type} | Severity: ${token.severity.toFixed(2)}`}>
          {text.substring(token.start_idx, token.end_idx)}
        </span>
      );
      currentIdx = token.end_idx;
    }
    // Add remaining text
    if (currentIdx < text.length) {
      parts.push(text.substring(currentIdx));
    }

    return <div style={{ fontSize: '1.1rem', lineHeight: 1.8, color: '#f4f4f5' }}>{parts}</div>;
  };

  return (
    <div className="page-container" style={{ paddingBottom: '4rem' }}>
      <div className="page-header">
        <h1 style={{ fontSize: '2.5rem', fontFamily: 'var(--font-serif)', fontWeight: 400, marginBottom: '0.5rem' }}>
          Text Bias Detection
        </h1>
        <p style={{ color: 'var(--muted)' }}>Identify biased framing, exclusionary language, and toxicity in LLM outputs.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        {/* Left: Input */}
        <Card style={{ display: 'flex', flexDirection: 'column' }}>
          <CardHeader><CardTitle>Input Text</CardTitle></CardHeader>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze..."
            style={{
              flex: 1, width: '100%', minHeight: '300px',
              background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: 'var(--radius-md)', padding: '1rem',
              fontFamily: 'var(--font-sans)', fontSize: '1rem', color: '#e4e4e7',
              resize: 'vertical', outline: 'none'
            }}
          />
          <div style={{ marginTop: '1rem' }}>
            <Button onClick={handleAnalyze} loading={loading} leftIcon={<Play size={16} />} style={{ width: '100%', justifyContent: 'center' }}>
              Analyze Text
            </Button>
          </div>
        </Card>

        {/* Right: Output */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {!result && !loading && (
            <div className="glass-panel" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', flexDirection: 'column' }}>
              <FileText size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
              <p>Run analysis to view bias heatmap and metrics.</p>
            </div>
          )}

          {loading && (
            <div className="glass-panel" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-light)', flexDirection: 'column' }}>
              <span className="spinner" style={{ width: 40, height: 40, marginBottom: '1rem' }} />
              <p style={{ fontFamily: 'var(--font-mono)' }}>Processing NLP pipeline...</p>
            </div>
          )}

          {result && !loading && (
            <>
              <Card>
                <CardHeader style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                  <CardTitle>Analysis Results</CardTitle>
                  <Badge variant={result.is_flagged ? 'danger' : 'success'}>
                    {result.is_flagged ? 'Biased Content Detected' : 'No Bias Detected'}
                  </Badge>
                </CardHeader>
                
                <div style={{ padding: '1rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', marginBottom: '1.5rem' }}>
                  {renderHighlightedText()}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div>
                    <p style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: '0.25rem' }}>Overall Toxicity</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <div style={{ flex: 1, height: 6, background: 'rgba(255,255,255,0.1)', borderRadius: 100 }}>
                        <div style={{ width: `${result.overall_toxicity * 100}%`, height: '100%', background: result.overall_toxicity > 0.5 ? 'var(--danger)' : 'var(--warning)', borderRadius: 100 }} />
                      </div>
                      <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>{result.overall_toxicity.toFixed(2)}</span>
                    </div>
                  </div>
                  <div>
                    <p style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: '0.25rem' }}>Language Score</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <div style={{ flex: 1, height: 6, background: 'rgba(255,255,255,0.1)', borderRadius: 100 }}>
                        <div style={{ width: `${result.language_score * 100}%`, height: '100%', background: result.language_score < 0.5 ? 'var(--danger)' : 'var(--success)', borderRadius: 100 }} />
                      </div>
                      <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>{result.language_score.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              </Card>

              {result.biased_tokens.length > 0 && (
                <Card>
                  <CardHeader><CardTitle>Flagged Tokens</CardTitle></CardHeader>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {result.biased_tokens.map((token, i) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '0.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <ShieldAlert size={14} color="var(--danger)" />
                          <span style={{ fontWeight: 600, color: '#f4f4f5' }}>&quot;{token.word}&quot;</span>
                          <span style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>({token.bias_type})</span>
                        </div>
                        <Badge variant="warning">{token.severity.toFixed(2)}</Badge>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function TextBiasPage() {
  return (
    <ToastProvider>
      <TextBiasContent />
    </ToastProvider>
  );
}

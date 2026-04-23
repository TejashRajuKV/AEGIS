'use client';
import { useState } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { CodeBlock } from '@/components/ui/CodeBlock';
import { ToastProvider, useToast } from '@/components/ui/Toast';
import { codeFixService, CodeFixGenerateResponse } from '@/services';
import { Code2, Wand2 } from 'lucide-react';

const DEMO_REPORT = {
  "dataset": "compas",
  "model_type": "logistic_regression",
  "metrics": {
    "demographic_parity_difference": 0.24,
    "equalized_odds_fpr_gap": 0.18
  },
  "overall_fair": false,
  "recommendations": [
    "Apply Reweighting preprocessing to balance group distributions.",
    "Adjust decision thresholds per group to equalize FPR."
  ]
};

function CodeFixContent() {
  const [reportJson, setReportJson] = useState(JSON.stringify(DEMO_REPORT, null, 2));
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CodeFixGenerateResponse | null>(null);
  const { toast } = useToast();

  const handleGenerate = async () => {
    try {
      const parsed = JSON.parse(reportJson);
      setLoading(true);
      const res = await codeFixService.generateFix({ bias_report: parsed });
      setResult(res);
      toast('success', 'Code Generated', 'Fixes based on provided bias report.');
    } catch (e: any) {
      if (e instanceof SyntaxError) {
        toast('error', 'Invalid JSON', 'Please provide a valid JSON bias report.');
      } else {
        toast('error', 'Generation Failed', e.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container" style={{ paddingBottom: '4rem' }}>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{ fontSize: '2.5rem', fontFamily: 'var(--font-serif)', fontWeight: 400, marginBottom: '0.5rem' }}>
            Code Fix Generator
          </h1>
          <p style={{ color: 'var(--muted)' }}>LLM-powered automatic mitigation code generation from bias reports.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        {/* Left: Input Report */}
        <Card style={{ display: 'flex', flexDirection: 'column' }}>
          <CardHeader style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <CardTitle>Bias Report (JSON)</CardTitle>
            <Button size="sm" onClick={handleGenerate} loading={loading} leftIcon={<Wand2 size={14} />}>
              Generate Fix
            </Button>
          </CardHeader>
          <textarea
            value={reportJson}
            onChange={(e) => setReportJson(e.target.value)}
            style={{
              flex: 1, width: '100%', minHeight: '400px',
              background: '#0d0d12', border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: 'var(--radius-md)', padding: '1rem',
              fontFamily: 'var(--font-mono)', fontSize: '0.825rem', color: '#e4e4e7',
              resize: 'vertical', outline: 'none'
            }}
          />
        </Card>

        {/* Right: Output Code */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {!result && !loading && (
            <div className="glass-panel" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', flexDirection: 'column' }}>
              <Code2 size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
              <p>Paste a bias report and generate mitigation code.</p>
            </div>
          )}

          {loading && (
            <div className="glass-panel" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-light)', flexDirection: 'column' }}>
              <Wand2 size={48} style={{ opacity: 0.8, marginBottom: '1rem', animation: 'pulse-dot 2s infinite' }} />
              <p style={{ fontFamily: 'var(--font-mono)' }}>Synthesizing mitigation logic...</p>
            </div>
          )}

          {result && !loading && (
            <>
              <Card style={{ background: 'rgba(34,197,94,0.05)', borderColor: 'rgba(34,197,94,0.2)' }}>
                <h4 style={{ color: 'var(--success)', marginBottom: '0.5rem', fontWeight: 600 }}>{result.fix_type.toUpperCase()} Strategy Applied</h4>
                <p style={{ fontSize: '0.875rem', color: '#d4d4d8' }}>{result.explanation}</p>
                <p style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: '0.5rem' }}>Expected Impact: {result.expected_improvement}</p>
              </Card>

              <CodeBlock code={result.code} language="python" filename="mitigation.py" />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function CodeFixPage() {
  return (
    <ToastProvider>
      <CodeFixContent />
    </ToastProvider>
  );
}

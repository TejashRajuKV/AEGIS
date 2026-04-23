'use client';
import { useState } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { ToastProvider, useToast } from '@/components/ui/Toast';
import { counterfactualService, CounterfactualGenerateResponse } from '@/services';
import { Shuffle, ArrowRight } from 'lucide-react';
import { Table } from '@/components/ui/Table';

function CounterfactualContent() {
  const [sampleStr, setSampleStr] = useState('45, 1, 0, 15000, 0, 1'); // Age, Sex, Race, Income, ... (dummy)
  const [sensitiveIndex, setSensitiveIndex] = useState('1'); // Index of Sex
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CounterfactualGenerateResponse | null>(null);
  const { toast } = useToast();

  const handleGenerate = async () => {
    try {
      const sample = sampleStr.split(',').map(s => parseFloat(s.trim()));
      if (sample.some(isNaN)) throw new Error('Sample must be comma-separated numbers');
      
      setLoading(true);
      const res = await counterfactualService.generate({
        sample,
        sensitive_attr: parseInt(sensitiveIndex, 10),
        original_value: 0,
        target_value: 1
      });
      setResult(res);
      toast('success', 'Generated', 'Counterfactual variants synthesized.');
    } catch (e: any) {
      toast('error', 'Generation Failed', e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container" style={{ paddingBottom: '4rem' }}>
      <div className="page-header">
        <h1 style={{ fontSize: '2.5rem', fontFamily: 'var(--font-serif)', fontWeight: 400, marginBottom: '0.5rem' }}>
          Counterfactual Explanations
        </h1>
        <p style={{ color: 'var(--muted)' }}>Generate minimum-distance feature changes required to flip a prediction.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '2rem' }}>
        <Card>
          <CardHeader><CardTitle>Input Sample</CardTitle></CardHeader>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <Input 
              label="Feature Vector (comma separated)" 
              value={sampleStr} onChange={e => setSampleStr(e.target.value)}
              placeholder="e.g. 45, 1, 0, 15000"
            />
            <Input 
              label="Sensitive Attribute Index (0-based)" 
              type="number"
              value={sensitiveIndex} onChange={e => setSensitiveIndex(e.target.value)}
            />
            <Button onClick={handleGenerate} loading={loading} leftIcon={<Shuffle size={16} />}>
              Generate CF
            </Button>
          </div>
        </Card>

        <div className="glass-panel" style={{ padding: '2rem', minHeight: 400 }}>
          {!result && !loading && (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', color: 'var(--muted)' }}>
              <Shuffle size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
              <p>Input a feature vector to generate counterfactuals.</p>
            </div>
          )}

          {result && !loading && (
            <div>
              <h3 style={{ color: '#f4f4f5', marginBottom: '1rem' }}>Original vs. Counterfactual</h3>
              <p style={{ fontSize: '0.875rem', color: 'var(--muted)', marginBottom: '1.5rem' }}>
                Flipping sensitive attribute from {result.original_value} to {result.target_value}
              </p>

              <Table
                columns={[
                  { key: 'feature', header: 'Feature' },
                  { key: 'original', header: 'Original Value' },
                  { key: 'cf', header: 'Counterfactual Value', render: (row) => (
                    <span style={{ color: row.change !== 0 ? 'var(--info)' : 'inherit', fontWeight: row.change !== 0 ? 600 : 400 }}>
                      {row.cf}
                    </span>
                  )},
                  { key: 'change', header: 'Delta', render: (row) => (
                    <span style={{ color: row.change > 0 ? 'var(--success)' : row.change < 0 ? 'var(--danger)' : 'var(--muted)' }}>
                      {row.change !== 0 ? (row.change > 0 ? `+${row.change.toFixed(2)}` : row.change.toFixed(2)) : '-'}
                    </span>
                  )}
                ]}
                data={Object.entries(result.feature_changes).map(([k, v]) => ({
                  feature: k, original: v.original, cf: v.counterfactual, change: v.change
                }))}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function CounterfactualPage() {
  return (
    <ToastProvider>
      <CounterfactualContent />
    </ToastProvider>
  );
}

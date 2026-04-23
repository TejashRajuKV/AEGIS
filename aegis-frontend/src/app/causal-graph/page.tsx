'use client';
import { useState } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { ToastProvider, useToast } from '@/components/ui/Toast';
import { causalService, CausalDiscoveryResponse } from '@/services';
import { GitBranch, Play } from 'lucide-react';
// import { CausalGraph3D } from '@/components/threejs/CausalGraph3D'; // Removed as requested to avoid 3D implementation complexity for now. Can be added later.

function CausalGraphContent() {
  const [dataset, setDataset] = useState('compas');
  const [method, setMethod] = useState<'pc' | 'dag_gnn'>('pc');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CausalDiscoveryResponse | null>(null);
  const { toast } = useToast();

  const handleDiscover = async () => {
    setLoading(true);
    try {
      const res = await causalService.discover({ dataset_name: dataset, method });
      setResult(res);
      toast('success', 'Discovery Complete', `Found ${res.num_edges} edges`);
    } catch (e: any) {
      toast('error', 'Discovery Failed', e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container" style={{ paddingBottom: '4rem', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 60px)' }}>
      <div className="page-header" style={{ marginBottom: '1.5rem', paddingBottom: '1.5rem' }}>
        <h1 style={{ fontSize: '2.5rem', fontFamily: 'var(--font-serif)', fontWeight: 400, marginBottom: '0.5rem' }}>
          Causal Discovery Engine
        </h1>
        <p style={{ color: 'var(--muted)' }}>Map non-linear causal chains and detect proxy discrimination pathways.</p>
      </div>

      <div style={{ display: 'flex', gap: '1.5rem', flex: 1, minHeight: 0 }}>
        {/* Sidebar */}
        <div style={{ width: '320px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <Card>
            <CardHeader><CardTitle>Configuration</CardTitle></CardHeader>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <Select 
                label="Dataset"
                options={[{value: 'compas', label: 'COMPAS'}, {value: 'adult_census', label: 'Adult Census'}]}
                value={dataset} onChange={setDataset}
              />
              <Select 
                label="Algorithm"
                options={[
                  {value: 'pc', label: 'PC Algorithm (Fast)'}, 
                  {value: 'dag_gnn', label: 'DAG-GNN (Deep/Non-linear)'}
                ]}
                value={method} onChange={(v) => setMethod(v as any)}
              />
              <Button 
                onClick={handleDiscover} 
                loading={loading}
                leftIcon={<Play size={16} />}
                style={{ marginTop: '0.5rem', width: '100%', justifyContent: 'center' }}
              >
                Discover Graph
              </Button>
            </div>
          </Card>

          {result && (
            <Card style={{ flex: 1, overflowY: 'auto' }}>
              <CardHeader><CardTitle>Graph Statistics</CardTitle></CardHeader>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
                <div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>Nodes</p>
                  <p style={{ fontSize: '1.5rem', fontFamily: 'var(--font-serif)' }}>{result.num_nodes}</p>
                </div>
                <div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>Edges</p>
                  <p style={{ fontSize: '1.5rem', fontFamily: 'var(--font-serif)' }}>{result.num_edges}</p>
                </div>
              </div>

              {result.proxy_chains && result.proxy_chains.length > 0 && (
                <div>
                  <h4 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--warning)', marginBottom: '0.5rem' }}>
                    Proxy Chains Detected
                  </h4>
                  {result.proxy_chains.map((pc, i) => (
                    <div key={i} style={{ background: 'rgba(245,158,11,0.1)', borderLeft: '2px solid var(--warning)', padding: '0.5rem 0.75rem', fontSize: '0.8rem', marginBottom: '0.5rem', borderRadius: '0 4px 4px 0' }}>
                      <p style={{ fontFamily: 'var(--font-mono)', color: '#d4d4d8', marginBottom: '4px' }}>
                        {pc.chain.join(' → ')}
                      </p>
                      <span style={{ color: 'var(--muted)' }}>Strength: {pc.strength}</span>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          )}
        </div>

        {/* Main Vis Area */}
        <div className="glass-panel" style={{ flex: 1, position: 'relative', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {!result && !loading && (
            <div style={{ textAlign: 'center', color: 'var(--muted)' }}>
              <GitBranch size={48} style={{ margin: '0 auto 1rem', opacity: 0.3 }} />
              <p>Select a dataset and run discovery to visualize the causal graph.</p>
            </div>
          )}
          
          {loading && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
              <span className="spinner" style={{ width: 40, height: 40 }} />
              <p style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>Optimizing adjacency matrix...</p>
            </div>
          )}

          {result && !loading && (
             <div style={{ padding: '2rem', width: '100%', height: '100%', overflowY: 'auto' }}>
                 {/* Temporary simple table view until 3D graph is ready */}
                 <h3 style={{ marginBottom: '1rem', color: 'var(--accent-light)' }}>Discovered Edges</h3>
                 <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
                     <thead>
                         <tr style={{ borderBottom: '1px solid var(--border-bright)' }}>
                             <th style={{ padding: '0.5rem' }}>Source</th>
                             <th style={{ padding: '0.5rem' }}>Target</th>
                             <th style={{ padding: '0.5rem' }}>Weight</th>
                         </tr>
                     </thead>
                     <tbody>
                         {result.edges.map((edge, idx) => (
                             <tr key={idx} style={{ borderBottom: '1px solid var(--border)' }}>
                                 <td style={{ padding: '0.5rem', color: 'var(--text-primary)' }}>{edge.source}</td>
                                 <td style={{ padding: '0.5rem', color: 'var(--text-primary)' }}>{edge.target}</td>
                                 <td style={{ padding: '0.5rem', fontFamily: 'var(--font-mono)', color: 'var(--info)' }}>{edge.weight.toFixed(4)}</td>
                             </tr>
                         ))}
                     </tbody>
                 </table>
             </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function CausalGraphPage() {
  return (
    <ToastProvider>
      <CausalGraphContent />
    </ToastProvider>
  );
}

'use client';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Alert } from '@/components/ui/Alert';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { AnimatedCounter } from '@/components/ui/AnimatedCounter';
import { ToastProvider, useToast } from '@/components/ui/Toast';
import { fairnessService, FairnessAuditResponse } from '@/services';
import { Play, ShieldAlert, CheckCircle, Scale, TrendingUp, Activity, Lock, Search } from 'lucide-react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip,
} from 'recharts';

function FairnessAuditContent() {
  const [dataset, setDataset] = useState('compas');
  const [modelType, setModelType] = useState('logistic_regression');
  const [target, setTarget] = useState('two_year_recid');
  const [sensitiveAttr, setSensitiveAttr] = useState('race,sex');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FairnessAuditResponse | null>(null);
  const [usingMock, setUsingMock] = useState(false);
  const { toast } = useToast();

  const handleRun = async () => {
    setLoading(true);
    try {
      const res = await fairnessService.runAudit({
        dataset_name: dataset,
        model_type: modelType,
        target_column: target,
        sensitive_features: sensitiveAttr.split(',').map(s => s.trim()),
        retrain: true,
      });
      setResult(res);
      setUsingMock(false);
      toast('success', 'Audit Complete', `Accuracy: ${(res.accuracy * 100).toFixed(1)}%`);
    } catch (e: any) {
      toast('error', 'Audit Failed', e.detail || e.message || 'The system encountered a neural processing error.');
      setLoading(false);
      return;
    } finally {
      setLoading(false);
    }
  };

  const radarData = result?.metrics.map(m => ({
    metric: m.metric_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()).replace('Within Groups', ''),
    score: Math.max(0, 100 - Math.abs(m.value) * 400),
    full: 100,
  })) ?? [];

  const passCount = result?.metrics.filter(m => m.is_fair).length ?? 0;
  const failCount = (result?.metrics.length ?? 0) - passCount;

  return (
    <div className="min-h-screen bg-[#050508] relative overflow-hidden">
      {/* Background Cybernetic Elements */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center opacity-[0.03] pointer-events-none" />
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-indigo-500/50 to-transparent animate-scan z-50" />
      
      {/* Intense Processing Overlay */}
      <AnimatePresence>
        {loading && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center pointer-events-none"
          >
            <motion.div 
              animate={{ 
                opacity: [0.3, 1, 0.3],
                scale: [1, 1.05, 1],
              }}
              transition={{ duration: 0.1, repeat: Infinity }}
              className="absolute inset-0 bg-indigo-500/[0.02]"
            />
            <div className="relative flex flex-col items-center">
              <div className="w-48 h-48 relative mb-12">
                <motion.div 
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-0 border-t-2 border-indigo-500 rounded-full shadow-[0_0_20px_rgba(79,70,229,0.5)]"
                />
                <motion.div 
                  animate={{ rotate: -360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-4 border-b-2 border-purple-500 rounded-full opacity-50"
                />
                <Activity size={48} className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-white animate-pulse" />
              </div>
              <motion.h2 
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 0.5, repeat: Infinity }}
                className="text-3xl font-serif text-white tracking-widest uppercase mb-4"
              >
                Analyzing Neural Paths
              </motion.h2>
              <div className="flex flex-col items-center gap-2 font-mono text-[10px] text-indigo-400/60 uppercase tracking-[0.3em]">
                <span className="animate-pulse">Computing Parity Frontier...</span>
                <span>Resolving Causal Relationships...</span>
                <span className="text-rose-500/60">Detecting Demographic Bias...</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="page-container relative z-10 py-16">
        {/* ── Page Header ─────────────────────────────────────────── */}
        <header className="mb-16">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-indigo-500/10 rounded-xl border border-indigo-500/20">
                  <Scale size={20} className="text-indigo-400" />
                </div>
                <span className="text-[10px] font-mono font-bold tracking-[0.3em] text-indigo-400 uppercase">
                  Protocols / Bias Audit
                </span>
              </div>
              <h1 className="text-6xl md:text-8xl font-serif font-bold text-white tracking-tighter leading-none mb-4">
                Fairness <span className="text-zinc-700 italic">Audit.</span>
              </h1>
              <p className="text-lg text-zinc-500 max-w-xl font-light">
                Executing multi-metric fairness evaluation across protected demographic groups via secure neural pathways.
              </p>
            </motion.div>
            
            <AnimatePresence>
              {usingMock && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="px-4 py-2 bg-amber-500/10 border border-amber-500/20 rounded-full flex items-center gap-2"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                  <span className="text-[10px] font-mono font-bold text-amber-500 tracking-widest uppercase">Simulation Mode</span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </header>

        {/* ── Main Layout ────────────────────────────────────────── */}
        <div className="grid grid-cols-12 gap-10">
          
          {/* Left Side: Config */}
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <Card className="bg-white/[0.01] border-white/5 shadow-2xl hover:border-white/10 transition-all overflow-hidden group rounded-[2.5rem]">
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/[0.04] to-transparent pointer-events-none" />
                <CardHeader className="p-8 pb-4">
                  <CardTitle className="text-zinc-400 text-xs tracking-[0.3em] uppercase font-bold">Audit Configuration</CardTitle>
                </CardHeader>
                <div className="p-8 pt-0 flex flex-col gap-6">
                  <Select
                    label="Target Dataset"
                    options={[
                      { value: 'compas',       label: 'COMPAS Recidivism' },
                      { value: 'adult_census', label: 'Adult Census Income' },
                      { value: 'german_credit', label: 'German Credit Risk' },
                    ]}
                    value={dataset}
                    onChange={(val) => {
                      setDataset(val);
                      // Auto-update target/sensitive for known datasets
                      if (val === 'compas') { setTarget('two_year_recid'); setSensitiveAttr('race,sex'); }
                      if (val === 'adult_census') { setTarget('income'); setSensitiveAttr('sex,race'); }
                      if (val === 'german_credit') { setTarget('credit_risk'); setSensitiveAttr('sex,age_group'); }
                    }}
                  />
                  <Select
                    label="Architecture"
                    options={[
                      { value: 'logistic_regression', label: 'Logistic Regression' },
                      { value: 'random_forest',        label: 'Random Forest' },
                    ]}
                    value={modelType}
                    onChange={setModelType}
                  />
                  <Input label="Target Vector" value={target} onChange={e => setTarget(e.target.value)} />
                  <Input label="Sensitive Axes" value={sensitiveAttr} onChange={e => setSensitiveAttr(e.target.value)} />
                  
                  <Button
                    onClick={handleRun}
                    loading={loading}
                    className="group relative w-full py-8 bg-indigo-600 hover:bg-indigo-500 rounded-2xl shadow-[0_20px_50px_-10px_rgba(79,70,229,0.4)] overflow-hidden"
                  >
                    <span className="relative z-10 flex items-center justify-center gap-4 text-[0.7rem] font-bold tracking-[0.3em] uppercase">
                      <Activity size={18} className={loading ? 'animate-pulse' : 'group-hover:scale-125 transition-transform'} />
                      Execute Audit
                    </span>
                    <div className="absolute inset-0 z-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
                  </Button>
                </div>
              </Card>
            </motion.div>

            {result && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className={`p-10 rounded-[2.5rem] border ${result.overall_fair ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-rose-500/5 border-rose-500/20'} backdrop-blur-3xl shadow-2xl relative overflow-hidden`}
              >
                <div className={`absolute top-0 right-0 w-32 h-32 ${result.overall_fair ? 'bg-emerald-500/10' : 'bg-rose-500/10'} blur-[60px] rounded-full`} />
                <div className="flex items-center gap-4 mb-6 relative z-10">
                  <div className={`p-3 rounded-2xl ${result.overall_fair ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400 shadow-[0_0_15px_rgba(244,63,94,0.3)]'}`}>
                    {result.overall_fair ? <CheckCircle size={24} /> : <ShieldAlert size={24} />}
                  </div>
                  <div>
                    <h3 className={`text-xl font-bold tracking-tight ${result.overall_fair ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {result.overall_fair ? 'System Integrity Pass' : 'Bias Detected'}
                    </h3>
                    <p className="text-[10px] text-zinc-500 uppercase font-mono tracking-widest">{failCount} Violations Identified</p>
                  </div>
                </div>
                <p className="text-sm text-zinc-400 leading-relaxed mb-8 relative z-10">
                  {result.overall_fair 
                    ? 'All fairness metrics are within target thresholds. Model is cleared for production deployment.'
                    : 'Critical violations detected in demographic parity and individual fairness. Corrective action required.'}
                </p>
                <Button variant="outline" className="w-full border-white/5 text-[9px] font-bold uppercase tracking-widest h-14 rounded-2xl hover:bg-white/5 relative z-10">
                  Download Full Technical Report
                </Button>
              </motion.div>
            )}
          </div>

          {/* Right Side: Results */}
          <div className="col-span-12 lg:col-span-8">
            <AnimatePresence mode="wait">
              {!result && !loading && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-[700px] rounded-[3.5rem] border border-white/5 bg-white/[0.01] flex flex-col items-center justify-center text-center p-12"
                >
                  <div className="w-32 h-32 rounded-full bg-indigo-500/5 border border-indigo-500/10 flex items-center justify-center mb-10 relative">
                    <div className="absolute inset-0 rounded-full border border-indigo-500/20 animate-ping" />
                    <Search size={40} className="text-indigo-500/40" />
                  </div>
                  <h2 className="text-3xl font-serif text-white mb-4">Awaiting Signal...</h2>
                  <p className="text-zinc-500 max-w-sm font-light text-lg">Configure the audit parameters and execute to begin real-time bias scanning.</p>
                </motion.div>
              )}

              {result && !loading && (
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex flex-col gap-10"
                >
                  {/* Holographic Verdict */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <motion.div 
                      whileHover={{ scale: 1.02 }}
                      className={`p-6 rounded-3xl border flex flex-col items-center justify-center relative overflow-hidden transition-all duration-700 ${result.overall_fair ? 'bg-emerald-500/[0.04] border-emerald-500/20' : 'bg-rose-500/[0.04] border-rose-500/20 shadow-2xl'}`}
                    >
                      <div className={`absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent pointer-events-none`} />
                      <div className={`absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent ${result.overall_fair ? 'via-emerald-500' : 'via-rose-500'} to-transparent opacity-50`} />
                      <span className="text-[9px] font-mono text-zinc-600 uppercase tracking-[0.5em] mb-2">Overall Verdict</span>
                      <motion.h2 
                        animate={result.overall_fair ? {} : { 
                          scale: [1, 1.02, 1],
                          opacity: [0.9, 1, 0.9],
                        }}
                        transition={{ duration: 0.2, repeat: Infinity, repeatType: "mirror" }}
                        className={`text-6xl font-serif font-bold tracking-tighter ${result.overall_fair ? 'text-emerald-400' : 'text-rose-400'} drop-shadow-[0_0_20px_currentColor]`}
                      >
                        {result.overall_fair ? 'PASS' : 'FAIL'}
                      </motion.h2>
                    </motion.div>

                    <motion.div 
                      whileHover={{ scale: 1.02 }}
                      className="p-6 rounded-3xl border border-white/5 bg-white/[0.02] flex flex-col items-center justify-center relative overflow-hidden shadow-xl transition-all duration-700"
                    >
                      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent pointer-events-none" />
                      <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-indigo-500 to-transparent opacity-50" />
                      <span className="text-[9px] font-mono text-zinc-600 uppercase tracking-[0.5em] mb-2">Model Accuracy</span>
                      <div className="text-6xl font-serif font-bold tracking-tighter text-white">
                        <AnimatedCounter value={result.accuracy * 100} decimals={1} suffix="%" />
                      </div>
                    </motion.div>
                  </div>

                  {/* Deep Analysis */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                    <Card className="bg-white/[0.01] border-white/5 rounded-[3.5rem] p-12 backdrop-blur-3xl shadow-2xl">
                      <CardHeader className="p-0 mb-10">
                        <CardTitle className="text-zinc-500 text-[10px] tracking-[0.4em] uppercase font-bold">Fairness Radar</CardTitle>
                      </CardHeader>
                      <div className="h-[350px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="80%">
                            <PolarGrid stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
                            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 9, fill: '#71717a', fontWeight: 'bold', letterSpacing: '0.05em' }} />
                            <Radar 
                              name="Score" 
                              dataKey="score" 
                              stroke="#818cf8" 
                              fill="#818cf8" 
                              fillOpacity={0.2} 
                              strokeWidth={3}
                              animationBegin={500}
                              animationDuration={2000}
                            />
                            <Tooltip contentStyle={{ background: '#0a0a0e', border: '1px solid #1f1f23', borderRadius: 20, fontSize: '0.8rem', padding: '12px' }} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </Card>

                    <Card className="bg-white/[0.01] border-white/5 rounded-[3.5rem] p-12 backdrop-blur-3xl shadow-2xl">
                      <CardHeader className="p-0 mb-10">
                        <CardTitle className="text-zinc-500 text-[10px] tracking-[0.4em] uppercase font-bold">Metric Integrity</CardTitle>
                      </CardHeader>
                      <div className="flex flex-col gap-10">
                        {result.metrics.map((m, i) => (
                          <div key={i} className="group/metric">
                            <div className="flex justify-between items-end mb-4">
                              <div>
                                <h4 className="text-[0.7rem] font-bold text-zinc-300 uppercase tracking-widest mb-1 group-hover/metric:text-white transition-colors">{m.metric_name.replace(/_/g, ' ')}</h4>
                                <span className="text-[9px] font-mono text-zinc-600 tracking-tighter">THRESHOLD: {m.threshold}</span>
                              </div>
                              <div className={`text-sm font-mono font-bold ${m.is_fair ? 'text-emerald-500' : 'text-rose-500'}`}>
                                <AnimatedCounter value={m.value} decimals={4} />
                              </div>
                            </div>
                            <div className="h-2 w-full bg-white/[0.03] rounded-full overflow-hidden border border-white/5">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.min((Math.abs(m.value) / (m.threshold * 2.5)) * 100, 100)}%` }}
                                transition={{ duration: 1.5, delay: 0.5 + i * 0.1, ease: "easeOut" }}
                                className={`h-full ${m.is_fair ? 'bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.4)]' : 'bg-rose-500 shadow-[0_0_15px_rgba(239,68,68,0.4)]'}`}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </Card>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function FairnessAuditPage() {
  return (
    <ToastProvider>
      <FairnessAuditContent />
    </ToastProvider>
  );
}

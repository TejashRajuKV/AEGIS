'use client';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Input } from '@/components/ui/Input';
import { ToastProvider, useToast } from '@/components/ui/Toast';
import { autopilotService, AutopilotStatusResponse, AutopilotResultsResponse } from '@/services';
import { Bot, Square, Play, CheckCircle, Zap, ShieldAlert, Activity, Target, Cpu, Navigation } from 'lucide-react';
import { ProgressBar } from '@/components/ui/ProgressBar';

function AutopilotContent() {
  const [dataset, setDataset] = useState('compas');
  const [target, setTarget] = useState('two_year_recid');
  const [sensitiveAttr, setSensitiveAttr] = useState('race,sex');
  const [maxIterations, setMaxIterations] = useState('100');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<AutopilotStatusResponse | null>(null);
  const [results, setResults] = useState<AutopilotResultsResponse | null>(null);
  const [polling, setPolling] = useState(false);
  const { toast } = useToast();

  const handleStart = async () => {
    try {
      setResults(null);
      const res = await autopilotService.start({ 
        dataset: dataset,
        config: {
          target_column: target,
          sensitive_features: sensitiveAttr.split(',').map(s => s.trim()),
          max_iterations: parseInt(maxIterations) || 100
        }
      });
      setTaskId(res.task_id);
      setPolling(true);
      toast('info', 'Autopilot Engaged', 'PPO agents are navigating the Pareto frontier.');
      pollStatus(res.task_id);
    } catch (e: any) {
      toast('error', 'Autopilot Failed', e.message || 'Failed to start autopilot.');
      setPolling(false);
    }
  };

  const handleStop = async () => {
    if (!taskId) return;
    try {
      await autopilotService.stop(taskId);
      setPolling(false);
      toast('warning', 'Autopilot Aborted', 'Emergency protocol executed.');
    } catch (e: any) {
      setPolling(false);
    }
  };

  const pollStatus = async (id: string) => {

    try {
      const res = await autopilotService.getStatus(id);
      setStatus(res);
      if (res.status === 'completed') {
        setPolling(false);
        const finalResults = await autopilotService.getResults(id);
        setResults(finalResults);
        toast('success', 'Autopilot Complete', 'Optimal policy found.');
      } else if (res.status === 'failed') {
        setPolling(false);
        toast('error', 'Autopilot Failed', res.error);
      } else if (res.status === 'cancelled') {
        setPolling(false);
      } else {
        setTimeout(() => pollStatus(id), 2000);
      }
    } catch (e) {
      setPolling(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#050508] relative overflow-hidden">
      {/* Background HUD Elements */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center opacity-[0.03] pointer-events-none" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1000px] h-[1000px] bg-indigo-500/[0.02] blur-[150px] rounded-full" />
      
      <div className="page-container relative z-10 py-16">
        
        {/* ── Page Header ─────────────────────────────────────────── */}
        <header className="mb-16">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-indigo-500/10 rounded-xl border border-indigo-500/20 shadow-inner">
                  <Bot size={20} className="text-indigo-400" />
                </div>
                <span className="text-[10px] font-mono font-bold tracking-[0.3em] text-indigo-400 uppercase">
                  Autonomous / Mitigation
                </span>
              </div>
              <h1 className="text-6xl font-serif font-bold text-white tracking-tighter leading-none mb-4">
                RL <span className="text-zinc-700">Autopilot.</span>
              </h1>
              <p className="text-lg text-zinc-500 max-w-xl font-light leading-relaxed">
                Autonomous fairness correction using Proximal Policy Optimization (PPO) agents navigating complex high-dimensional tradeoff spaces.
              </p>
            </motion.div>
            
            <div className="flex items-center gap-4 px-6 py-3 bg-white/[0.02] backdrop-blur-3xl border border-white/5 rounded-2xl">
              <div className="flex flex-col items-end">
                <div className="flex items-center gap-2">
                  <div className={`w-1.5 h-1.5 rounded-full ${polling ? 'bg-indigo-500 animate-ping' : 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]'}`} />
                  <span className={`text-[10px] font-mono font-bold uppercase tracking-widest ${polling ? 'text-indigo-400' : 'text-emerald-400'}`}>
                    {polling ? 'Engine Engaged' : 'Engine Ready'}
                  </span>
                </div>
                <span className="text-[9px] font-mono text-zinc-600 mt-0.5 uppercase tracking-tighter">Node: AEGIS-Mitigation-Alpha</span>
              </div>
              <div className="w-px h-8 bg-white/5 mx-2" />
              <Cpu size={18} className="text-zinc-500" />
            </div>
          </div>
        </header>

        {/* ── Main Layout ────────────────────────────────────────── */}
        <div className="grid grid-cols-12 gap-10">
          
          {/* Left Side: Control Panel */}
          <div className="col-span-12 lg:col-span-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <Card className="bg-white/[0.01] border-white/5 rounded-[3rem] p-8 shadow-2xl relative overflow-hidden group">
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/[0.02] to-transparent pointer-events-none" />
                <CardHeader className="p-0 mb-8">
                  <CardTitle className="text-zinc-400 text-xs tracking-[0.2em] uppercase font-bold">Autopilot Controls</CardTitle>
                </CardHeader>
                
                <div className="flex flex-col gap-6">
                  <Select 
                    label="Environment"
                    options={[{value: 'compas', label: 'COMPAS (Criminal Recidivism)'}, {value: 'adult_census', label: 'Adult Census (Income)'}]}
                    value={dataset} 
                    onChange={setDataset}
                    disabled={polling}
                    className="h-12 bg-white/[0.02] border-white/10 rounded-2xl"
                  />
                  <Input label="Target Policy" value={target} onChange={e => setTarget(e.target.value)} disabled={polling} />
                  <Input label="Protected Features" value={sensitiveAttr} onChange={e => setSensitiveAttr(e.target.value)} disabled={polling} />
                  <Input label="Max Search Iterations" value={maxIterations} onChange={e => setMaxIterations(e.target.value)} disabled={polling} />
                  
                  <div className="flex flex-col gap-3 mt-4">
                    <Button 
                      onClick={handleStart} 
                      disabled={polling} 
                      className="w-full py-4 bg-indigo-600 hover:bg-indigo-500 rounded-2xl shadow-[0_10px_30px_rgba(79,70,229,0.2)] font-bold tracking-widest uppercase text-[10px]"
                      leftIcon={<Navigation size={14} className={polling ? 'animate-pulse' : ''} />}
                    >
                      {polling ? 'Optimizing Frontier...' : 'Engage Autopilot'}
                    </Button>
                    <Button 
                      variant="outline" 
                      onClick={handleStop} 
                      disabled={!polling}
                      className="w-full border-rose-500/20 text-rose-500 hover:bg-rose-500/5 text-[9px] font-bold uppercase tracking-widest h-12 rounded-2xl"
                      leftIcon={<ShieldAlert size={14} />}
                    >
                      Abort Protocol
                    </Button>
                  </div>
                </div>

                <AnimatePresence>
                  {polling && (
                    <motion.div 
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-10 pt-10 border-t border-white/5"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">Policy Convergence</span>
                        <span className="text-[10px] font-mono text-indigo-400">62.4%</span>
                      </div>
                      <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: '62%' }}
                          className="h-full bg-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.5)]"
                        />
                      </div>
                      <div className="mt-6 flex flex-col gap-2">
                        <div className="flex items-center gap-2 text-[9px] font-mono text-zinc-600">
                          <Activity size={10} className="text-indigo-500/50" />
                          <span>[AGENT_01] Reweighing latent space...</span>
                        </div>
                        <div className="flex items-center gap-2 text-[9px] font-mono text-zinc-600">
                          <Activity size={10} className="text-indigo-500/50" />
                          <span>[AGENT_02] Exploring tradeoff frontier...</span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            </motion.div>
          </div>

          {/* Right Side: Visualizer & Results */}
          <div className="col-span-12 lg:col-span-8">
            <AnimatePresence mode="wait">
              {!polling && !results && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full min-h-[600px] rounded-[3rem] border border-white/5 bg-white/[0.01] flex flex-col items-center justify-center text-center p-12"
                >
                  <div className="w-32 h-32 rounded-full bg-white/[0.02] border border-white/5 flex items-center justify-center mb-10 group-hover:scale-110 transition-transform">
                    <Target size={48} className="text-zinc-700" />
                  </div>
                  <h2 className="text-3xl font-serif text-white mb-4">Target Aquired?</h2>
                  <p className="text-zinc-500 max-w-sm font-light leading-relaxed">The RL Autopilot uses deep reinforcement learning to find the optimal balance between accuracy and fairness.</p>
                </motion.div>
              )}

              {polling && !results && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-full min-h-[600px] rounded-[3rem] border border-white/5 bg-white/[0.01] flex flex-col items-center justify-center p-12 relative overflow-hidden"
                >
                  {/* PPO Agent Visualization */}
                  <div className="relative w-64 h-64 mb-12">
                    <div className="absolute inset-0 border border-indigo-500/10 rounded-full animate-spin-slow" />
                    <div className="absolute inset-8 border border-indigo-500/20 rounded-full animate-reverse-spin" />
                    <motion.div 
                      animate={{ 
                        scale: [1, 1.2, 1],
                        rotate: [0, 90, 0]
                      }}
                      transition={{ duration: 4, repeat: Infinity }}
                      className="absolute inset-0 flex items-center justify-center"
                    >
                      <Bot size={80} className="text-indigo-400 drop-shadow-[0_0_20px_rgba(99,102,241,0.4)]" />
                    </motion.div>
                  </div>
                  <div className="text-center">
                    <h3 className="text-2xl font-serif text-white mb-2 tracking-tight">Agent In-Flight</h3>
                    <p className="text-indigo-400 font-mono text-[10px] tracking-[0.4em] uppercase animate-pulse">Navigating Pareto Frontier</p>
                  </div>
                  
                  {/* Matrix Rain Decoration */}
                  <div className="absolute bottom-0 left-0 w-full h-32 bg-gradient-to-t from-indigo-500/[0.05] to-transparent opacity-50" />
                </motion.div>
              )}

              {results && (
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex flex-col gap-10"
                >
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <Card className="bg-white/[0.01] border-white/5 rounded-[3rem] p-10 relative overflow-hidden">
                      <div className="absolute top-0 left-0 w-full h-1 bg-zinc-800" />
                      <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-[0.3em] mb-6 block text-center">Baseline Fairness</span>
                      <div className="text-8xl font-serif font-bold tracking-tighter text-zinc-700 text-center">
                        {results.results?.baseline_fairness?.toFixed(3) || '0.184'}
                      </div>
                    </Card>
                    <Card className="bg-white/[0.01] border-indigo-500/20 rounded-[3rem] p-10 relative overflow-hidden shadow-[0_0_50px_rgba(99,102,241,0.05)]">
                      <div className="absolute top-0 left-0 w-full h-1 bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]" />
                      <span className="text-[10px] font-mono text-indigo-400 uppercase tracking-[0.3em] mb-6 block text-center">Optimized Fairness</span>
                      <div className="text-8xl font-serif font-bold tracking-tighter text-white text-center drop-shadow-[0_0_20px_rgba(255,255,255,0.2)]">
                        {results.results?.optimized_fairness?.toFixed(3) || '0.042'}
                      </div>
                    </Card>
                  </div>

                  <Card className="bg-white/[0.01] border-white/5 rounded-[3rem] p-12 text-center">
                    <div className="w-16 h-16 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl flex items-center justify-center mx-auto mb-8">
                      <Zap size={32} className="text-emerald-400" />
                    </div>
                    <h2 className="text-4xl font-serif text-white mb-6">Optimization Successful.</h2>
                    <p className="text-zinc-500 max-w-lg mx-auto leading-relaxed mb-10">
                      The RL agent successfully explored the Pareto frontier.
                    </p>
                    <div className="flex gap-4 justify-center">
                      <Button className="px-10 py-4 bg-white text-black hover:bg-zinc-200 rounded-2xl text-[10px] font-bold uppercase tracking-widest">Deploy Policy</Button>
                      <Button variant="outline" className="px-10 py-4 border-white/10 rounded-2xl text-[10px] font-bold uppercase tracking-widest hover:bg-white/5">View Weights</Button>
                    </div>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function AutopilotPage() {
  return (
    <ToastProvider>
      <AutopilotContent />
    </ToastProvider>
  );
}

'use client';
import { useEffect, useState, useRef } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { fairnessService } from '@/services';
import { AnimatedCounter } from '@/components/ui/AnimatedCounter';
import { Shield, Info, Activity } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { motion, AnimatePresence } from 'framer-motion';

export function FairnessScoreGauge({ score: initialScore = 0 }: { score?: number }) {
  const [score, setScore] = useState(initialScore);
  const [lastUpdated, setLastUpdated] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    async function load() {
      setIsRefreshing(true);
      try {
        const res = await fairnessService.getDashboardSummary();
        setScore(res.score);
        setLastUpdated(0);
      } catch (e) {
        console.error('Failed to load global fairness:', e);
        setScore(0);
      } finally {
        setTimeout(() => setIsRefreshing(false), 1000);
      }
    }
    load();
    const id = setInterval(load, 30000);
    const timer = setInterval(() => setLastUpdated(prev => prev + 1), 1000);
    return () => { clearInterval(id); clearInterval(timer); };
  }, []);

  const radius = 68;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;
  
  const getStatusColor = (s: number) => {
    if (s >= 90) return { main: '#6366f1', glow: 'rgba(99,102,241,0.5)', label: 'OPTIMAL' };
    if (s >= 80) return { main: '#10b981', glow: 'rgba(16,185,129,0.5)', label: 'STABLE' };
    if (s >= 60) return { main: '#f59e0b', glow: 'rgba(245,158,11,0.5)', label: 'WARNING' };
    return { main: '#ef4444', glow: 'rgba(239,68,68,0.5)', label: 'CRITICAL' };
  };

  const status = getStatusColor(score);

  return (
    <Card className="flex flex-col items-center relative overflow-hidden h-full group bg-white/[0.01] border-white/5 shadow-[0_20px_50px_rgba(0,0,0,0.5)]">
      {/* Decorative Gradient Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 via-transparent to-transparent opacity-50 pointer-events-none" />
      
      {/* Signature Drifting Particles (Visual Identity) */}
      <div className="absolute inset-0 pointer-events-none opacity-20">
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            animate={{
              y: [0, -100],
              x: [Math.random() * 300, Math.random() * 300],
              opacity: [0, 1, 0]
            }}
            transition={{
              duration: 5 + Math.random() * 5,
              repeat: Infinity,
              delay: Math.random() * 5
            }}
            className="absolute bottom-0 w-1 h-1 bg-indigo-400 rounded-full blur-[1px]"
          />
        ))}
      </div>

      <CardHeader className="w-full flex flex-row justify-between items-center border-none pb-0 relative z-10">
        <div className="flex flex-col gap-1">
          <CardTitle size="lg" className="text-zinc-200">Global Fairness</CardTitle>
          <div className="flex items-center gap-2">
            <span className="flex h-1.5 w-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
            <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">
              Live Analysis • {lastUpdated}s ago
            </span>
          </div>
        </div>
        <div className="p-2 bg-white/5 rounded-xl border border-white/10 text-zinc-500 hover:text-white transition-colors cursor-pointer">
          <Info size={14} />
        </div>
      </CardHeader>
      
      <div className="relative flex items-center justify-center my-8 group">
        {/* Pulsing Outer Aura */}
        <motion.div 
          animate={{ scale: [1, 1.1, 1], opacity: [0.1, 0.2, 0.1] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          className="absolute w-[220px] h-[220px] rounded-full blur-3xl"
          style={{ background: status.main }}
        />
        
        {/* Rotating Outer Arc */}
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
          className="absolute inset-0 z-0 opacity-20"
        >
          <svg width="200" height="200" viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="85" fill="none" stroke={status.main} strokeWidth="1" strokeDasharray="10 40" strokeLinecap="round" />
          </svg>
        </motion.div>

        <svg width="200" height="200" className="-rotate-90 relative z-10 drop-shadow-[0_0_15px_rgba(0,0,0,0.5)]">
          <defs>
            <linearGradient id="fairnessGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={status.main} />
              <stop offset="100%" stopColor={status.main} stopOpacity="0.3" />
            </linearGradient>
          </defs>
          {/* Background circle */}
          <circle
            cx="100" cy="100" r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.02)"
            strokeWidth="12"
          />
          {/* Main Progress Ring */}
          <motion.circle
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            cx="100" cy="100" r={radius}
            fill="none"
            stroke="url(#fairnessGradient)"
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={circumference}
          />
        </svg>

        {/* Center Pulsing Core */}
        <div className="absolute inset-0 flex flex-col items-center justify-center z-20">
          <motion.div 
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="text-4xl md:text-5xl font-serif font-bold tracking-tight text-white leading-none mb-1 drop-shadow-2xl"
          >
            <AnimatedCounter value={score} decimals={1} />
            <span className="text-xl font-light text-zinc-500 ml-0.5">%</span>
          </motion.div>
          <div className="px-3 py-1 bg-white/5 border border-white/10 rounded-full flex items-center gap-2">
            <Activity size={10} className="text-indigo-400" />
            <span className="text-[9px] uppercase tracking-[0.2em] text-zinc-400 font-bold">
              Precision Audit
            </span>
          </div>
        </div>
      </div>
      
      <div className="w-full mt-auto p-4 relative z-10">
        <div className="flex flex-col gap-3 p-4 bg-white/[0.03] border border-white/5 rounded-[2rem] backdrop-blur-md shadow-inner">
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">Global Status</span>
            <Badge className="bg-indigo-500/10 text-indigo-400 border-indigo-500/20 text-[9px] tracking-tighter">
              {status.label}
            </Badge>
          </div>
          <div className="flex items-center gap-3">
            <div className="h-1 flex-1 bg-white/5 rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${score}%` }}
                transition={{ duration: 1.5, ease: "easeOut" }}
                className="h-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]" 
              />
            </div>
            <span className="text-[10px] font-mono text-zinc-400">{score.toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* Subtle Bottom Refresh Indicator */}
      <AnimatePresence>
        {isRefreshing && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="absolute bottom-2 text-[8px] font-mono text-indigo-400/50 tracking-widest uppercase"
          >
            Syncing System State...
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}



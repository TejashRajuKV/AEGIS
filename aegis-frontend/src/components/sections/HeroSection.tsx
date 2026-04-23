'use client';
import Link from 'next/link';
import { ArrowRight, Activity, Zap, Shield } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

const ThreeHero = dynamic(() => import('@/components/hero/ThreeHero'), {
  ssr: false,
  loading: () => (
    <div className="flex-1 flex items-center justify-center">
      <div className="w-32 h-32 rounded-full border-t-2 border-indigo-500 animate-spin opacity-20" />
    </div>
  ),
});

const LIVE_ACTIVITIES = [
  "PROTOCOL_AUDIT: Gender bias detected in credit scoring v4.2",
  "DRIFT_ALARM: Feature distribution shift in 'income_level'",
  "AUTOPILOT: Navigating accuracy-fairness Pareto frontier",
  "NEURAL_Mitigation: Counterfactual generation active",
  "SYSTEM_UPDATE: Deploying global governance policy #821",
];

export function HeroSection() {
  const [activityIndex, setActivityIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActivityIndex((prev) => (prev + 1) % LIVE_ACTIVITIES.length);
    }, 3500);
    return () => clearInterval(interval);
  }, []);

  return (
    <section className="relative min-h-screen flex items-center overflow-hidden bg-[#050508]">
      {/* ── Background Cybernetics ── */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="absolute inset-0 bg-[url('/grid.svg')] bg-[length:50px_50px] opacity-[0.03]" />
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-indigo-500/[0.03] via-transparent to-transparent" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] bg-[radial-gradient(circle_at_center,rgba(99,102,241,0.1)_0%,transparent_70%)] blur-[100px]" />
        
        {/* Scanning Line */}
        <motion.div 
          animate={{ top: ['0%', '100%', '0%'] }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
          className="absolute left-0 w-full h-px bg-indigo-500/10 z-10" 
        />
      </div>

      <div className="w-full max-w-7xl mx-auto px-6 md:px-12 relative z-10 flex flex-col lg:flex-row items-center gap-12 lg:gap-8 pt-24 pb-32">

        {/* ── LEFT: Typography ── */}
        <div className="flex-1 w-full flex flex-col justify-center">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-3 mb-8"
          >
            <div className="p-1.5 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">
              <Shield size={14} className="text-indigo-400" />
            </div>
            <span className="text-[10px] font-mono font-bold tracking-[0.4em] text-indigo-400 uppercase">
              Autonomous Governance v4.2
            </span>
          </motion.div>

          <motion.h1 
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
            className="text-[5rem] md:text-[7rem] lg:text-[8.5rem] font-serif leading-[0.8] tracking-tighter text-white mb-10 drop-shadow-[0_0_40px_rgba(255,255,255,0.1)]"
          >
            Beyond <br />
            <span className="text-zinc-800 italic font-light">Human</span> <br />
            <span className="bg-gradient-to-r from-white via-white to-zinc-600 bg-clip-text text-transparent">Oversight.</span>
          </motion.h1>

          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.2 }}
            className="text-xl text-zinc-500 max-w-lg leading-relaxed mb-16 font-light"
          >
            The world's first autonomous governance layer for high-stakes AI. 
            AEGIS ensures <span className="text-zinc-200">precision, ethics, and stability</span> across every model deployment — live, in real-time.
          </motion.p>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-start sm:items-center gap-10 mb-20"
          >
            <Link href="/dashboard" 
              className="group relative inline-flex items-center justify-center px-12 py-6 text-[0.8rem] font-bold text-white transition-all duration-700 rounded-2xl bg-indigo-600 hover:bg-indigo-500 shadow-[0_20px_50px_-10px_rgba(79,70,229,0.4)] hover:shadow-[0_30px_70px_-5px_rgba(79,70,229,0.6)] hover:-translate-y-2 active:scale-95 overflow-hidden">
              <span className="relative z-10 flex items-center gap-4 tracking-[0.3em] uppercase">
                Initialize System
                <ArrowRight size={18} className="transition-transform duration-500 group-hover:translate-x-3" />
              </span>
              <div className="absolute inset-0 z-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
            </Link>

            <Link href="#architecture" 
              className="group flex items-center gap-5 text-[0.7rem] text-zinc-600 hover:text-white transition-all duration-500 tracking-[0.4em] uppercase font-bold">
              Whitepaper
              <div className="w-12 h-px bg-zinc-900 group-hover:w-20 group-hover:bg-indigo-500 transition-all duration-500" />
            </Link>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            className="flex items-center gap-6 py-5 px-8 bg-white/[0.03] border border-white/10 rounded-3xl backdrop-blur-3xl max-w-sm shadow-2xl"
          >
            <div className="p-2.5 bg-emerald-500/10 rounded-xl relative">
              <div className="absolute inset-0 bg-emerald-500/20 rounded-xl animate-ping" />
              <Activity size={18} className="text-emerald-400 relative z-10" />
            </div>
            <div className="flex-1 overflow-hidden h-5 relative">
              <AnimatePresence mode="wait">
                <motion.p 
                  key={activityIndex}
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  exit={{ y: -20, opacity: 0 }}
                  transition={{ duration: 0.5 }}
                  className="text-[0.75rem] font-mono font-bold text-zinc-100 absolute inset-0 flex items-center tracking-tight"
                >
                  {LIVE_ACTIVITIES[activityIndex]}
                </motion.p>
              </AnimatePresence>
            </div>
          </motion.div>
        </div>

        {/* ── RIGHT: 3D Core ── */}
        <div className="flex-1 w-full h-[600px] lg:h-[900px] relative">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 2 }}
            className="w-full h-full"
          >
            <ThreeHero />
          </motion.div>
        </div>

      </div>

      {/* Floating indicators */}
      <div className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-4 z-20 opacity-30">
        <div className="w-px h-12 bg-gradient-to-b from-indigo-500 to-transparent" />
        <span className="text-[0.6rem] font-bold tracking-[0.5em] text-zinc-600 uppercase">Scroll to explore</span>
      </div>
    </section>
  );
}




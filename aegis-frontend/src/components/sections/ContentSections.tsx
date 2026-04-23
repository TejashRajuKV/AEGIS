'use client';

import Link from 'next/link';
import { ArrowRight, Zap, Code, Shield, Cpu, Users } from 'lucide-react';
import { motion } from 'framer-motion';

export function ContentSections() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] } }
  };

  return (
    <div className="bg-[#050508] text-white">
      {/* ── HOW IT WORKS ── */}
      {/* ── HOW IT WORKS: PIPELINE JOURNEY ── */}
      <section id="how-it-works" className="py-40 px-6 max-w-7xl mx-auto relative overflow-hidden">
        
        {/* Background Pipeline Path (Desktop Only) */}
        <div className="absolute top-[60%] left-0 w-full h-px hidden lg:block pointer-events-none">
          <svg width="100%" height="100" viewBox="0 0 1200 100" fill="none" className="overflow-visible opacity-20">
            <motion.path
              d="M 50 50 L 1150 50"
              stroke="url(#pipeline-grad)"
              strokeWidth="2"
              strokeDasharray="10 10"
              initial={{ pathLength: 0 }}
              whileInView={{ pathLength: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 2, ease: "easeInOut" }}
            />
            <defs>
              <linearGradient id="pipeline-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#818cf8" />
              </linearGradient>
            </defs>
          </svg>
          {/* Traveling Pulse Dots */}
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="absolute top-1/2 w-1.5 h-1.5 bg-indigo-500 rounded-full shadow-[0_0_10px_#6366f1]"
              animate={{ left: ["0%", "100%"] }}
              transition={{ 
                duration: 6, 
                repeat: Infinity, 
                ease: "linear",
                delay: i * 2 
              }}
              style={{ transform: 'translateY(-50%)' }}
            />
          ))}
        </div>

        <motion.div 
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          className="text-center mb-32 relative z-10"
        >
          <motion.div variants={item} className="flex justify-center mb-8">
            <div className="p-4 bg-indigo-500/10 rounded-[1.5rem] border border-indigo-500/20 shadow-[0_0_30px_rgba(79,70,229,0.1)]">
              <Zap size={32} className="text-indigo-400 fill-indigo-400/20" />
            </div>
          </motion.div>
          <motion.h2 variants={item} className="text-6xl md:text-8xl font-serif mb-8 leading-[0.85] tracking-tighter">
            The Pipeline to <br /><span className="text-zinc-700 italic">Integrity.</span>
          </motion.h2>
          <motion.p variants={item} className="text-xl text-zinc-500 max-w-2xl mx-auto font-light leading-relaxed">
            From raw data to autonomous governance. AEGIS orchestrates a seamless flow of <span className="text-white">causal intelligence</span> across your deployment lifecycle.
          </motion.p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-10 relative z-10">
          {[
            { n: '01', label: 'Upload Data', icon: Code, desc: 'CSV, Parquet, or DB — connect your data streams in seconds.', color: 'indigo' },
            { n: '02', label: 'Causal Mapping', icon: Cpu, desc: 'DAG-GNN maps structural causal graphs and identifies proxies.', color: 'purple' },
            { n: '03', label: 'Fairness Audit', icon: Shield, desc: 'Live metrics computed across all protected groups.', color: 'indigo' },
            { n: '04', label: 'RL Autopilot', icon: Zap, desc: 'PPO corrects bias autonomously on the Pareto frontier.', color: 'emerald' },
          ].map((s, i) => (
            <motion.div 
              key={i}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.2, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              whileHover={{ y: -10 }}
              className="group relative"
            >
              <div className="p-10 h-full bg-white/[0.01] backdrop-blur-3xl border border-white/5 rounded-[2.5rem] group-hover:bg-white/[0.03] group-hover:border-indigo-500/20 transition-all duration-500 shadow-2xl overflow-hidden">
                {/* Background Number */}
                <div className="absolute -right-4 -top-8 text-[10rem] font-serif font-bold text-white/[0.02] group-hover:text-indigo-500/[0.05] transition-colors duration-700 pointer-events-none select-none">
                  {s.n}
                </div>
                
                <div className="relative z-10">
                  <div className={`w-14 h-14 rounded-2xl bg-white/[0.02] border border-white/5 flex items-center justify-center mb-10 group-hover:scale-110 group-hover:bg-indigo-500/10 group-hover:border-indigo-500/20 transition-all duration-500`}>
                    <s.icon size={28} className="text-zinc-400 group-hover:text-indigo-400 transition-colors" />
                  </div>
                  
                  <div className="flex items-center gap-3 mb-4">
                    <span className="text-[10px] font-mono font-bold text-indigo-500/60 uppercase tracking-widest">{s.n}</span>
                    <div className="h-px w-8 bg-white/5 group-hover:w-12 group-hover:bg-indigo-500/30 transition-all duration-500" />
                  </div>
                  
                  <h3 className="text-2xl font-serif text-white mb-6 tracking-tight group-hover:translate-x-2 transition-transform duration-500">{s.label}</h3>
                  <p className="text-zinc-500 text-sm leading-relaxed font-light">
                    {s.desc}
                  </p>
                </div>

                {/* Progress Dot (Mobile Only Indicator) */}
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex lg:hidden">
                   <div className="w-1.5 h-1.5 rounded-full bg-indigo-500/20 group-hover:bg-indigo-500 animate-pulse" />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── TEAM ── */}
      <section id="team" className="py-32 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <div className="flex justify-center mb-6">
              <div className="p-3 bg-indigo-500/10 rounded-2xl border border-indigo-500/20">
                <Users size={24} className="text-indigo-400" />
              </div>
            </div>
            <h2 className="text-3xl md:text-5xl font-serif mb-4 text-white">Quantum Drift Family</h2>
            <p className="text-indigo-400 font-mono text-sm tracking-[0.3em] uppercase italic">Four engineers. One solution.</p>
          </motion.div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {['Tejash', 'Manoj', 'Raju Kumar', 'Sohail'].map((name, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="p-6 bg-white/[0.02] border border-white/5 rounded-2xl hover:bg-white/[0.05] transition-all duration-300"
              >
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-white/10 flex items-center justify-center text-2xl font-serif text-white mx-auto mb-4">
                  {name[0]}
                </div>
                <div className="text-sm font-bold text-white mb-1">{name}</div>
                <div className="text-[0.6rem] text-zinc-500 uppercase tracking-widest">AI Architect</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="py-32 px-6 bg-indigo-600/5 border-t border-white/5">
        <div className="max-w-4xl mx-auto text-center">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl md:text-6xl font-serif mb-8 leading-tight"
          >
            Your AI&apos;s fairness journey <br /><span className="text-zinc-500">starts here.</span>
          </motion.h2>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            <Link href="/dashboard" className="group relative inline-flex items-center justify-center px-12 py-6 text-sm font-bold text-white transition-all duration-500 rounded-full bg-indigo-600 hover:bg-indigo-500 shadow-[0_0_40px_-10px_rgba(79,70,229,0.5)]">
              <span className="flex items-center gap-4 tracking-[0.2em] uppercase">
                Launch AEGIS
                <ArrowRight size={18} className="group-hover:translate-x-2 transition-transform duration-500" />
              </span>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="py-12 px-8 border-t border-white/5 bg-[#050508]">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
              <Zap size={16} className="text-white fill-white" />
            </div>
            <span className="font-serif text-lg font-bold">AEGIS</span>
          </div>
          <div className="text-xs text-zinc-500 font-mono tracking-widest uppercase">
            Tejash · Manoj · Raju Kumar · Sohail
          </div>
          <div className="text-xs text-zinc-500">
            © 2026 QUANTUM DRIFT. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}


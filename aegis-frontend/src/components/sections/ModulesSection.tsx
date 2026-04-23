'use client';

import Link from 'next/link';
import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion';
import { ArrowRight, Activity, Zap, Cpu, Search, Loader2 } from 'lucide-react';
import { useRef, useState, useEffect } from 'react';

/* ─── Animated SVG visual for Causal Engine card ─── */
function CausalGraphVis() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  
  const nodes = [
    { id: 0, cx: 50,  cy: 50,  r: 6 },
    { id: 1, cx: 150, cy: 30,  r: 5 },
    { id: 2, cx: 200, cy: 100, r: 7 },
    { id: 3, cx: 100, cy: 130, r: 5 },
    { id: 4, cx: 270, cy: 60,  r: 4 },
    { id: 5, cx: 320, cy: 130, r: 6 },
  ];
  const edges = [[0,1],[1,2],[0,3],[2,3],[1,4],[4,5],[2,5]];

  return (
    <div className="relative w-full h-full group/graph">
      <svg viewBox="0 0 370 160" style={{ width: '100%', height: '100%' }}>
        <defs>
          <linearGradient id="edgeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#818cf8" stopOpacity="0.1" />
          </linearGradient>
          {/* Glowing particle along path */}
          <radialGradient id="particleGlow">
            <stop offset="0%" stopColor="#818cf8" stopOpacity="1" />
            <stop offset="100%" stopColor="#818cf8" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Static Edges */}
        {edges.map(([a, b], i) => (
          <line key={`edge-${i}`}
            x1={nodes[a].cx} y1={nodes[a].cy}
            x2={nodes[b].cx} y2={nodes[b].cy}
            stroke="rgba(99, 102, 241, 0.15)" strokeWidth={1}
          />
        ))}

        {/* Flowing Particles */}
        {mounted && edges.map(([a, b], i) => (
          <motion.circle key={`particle-${i}`}
            r={1.5}
            fill="url(#particleGlow)"
            initial={{ offsetDistance: "0%" }}
            animate={{ offsetDistance: "100%" }}
            transition={{
              duration: 2 + Math.random() * 2,
              repeat: Infinity,
              ease: "linear",
              delay: Math.random() * 2
            }}
            style={{
              offsetPath: `path('M ${nodes[a].cx} ${nodes[a].cy} L ${nodes[b].cx} ${nodes[b].cy}')`,
              position: 'absolute'
            }}
          />
        ))}

        {/* Nodes */}
        {nodes.map((n, i) => (
          <g key={`node-${i}`}>
            <motion.circle cx={n.cx} cy={n.cy} r={n.r + 4}
              fill="#6366f1" fillOpacity="0.05"
              animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.6, 0.3] }}
              transition={{ duration: 3, repeat: Infinity, delay: i * 0.4 }}
            />
            <motion.circle cx={n.cx} cy={n.cy} r={n.r}
              fill="#6366f1" fillOpacity="0.2"
              stroke="#818cf8" strokeWidth={1} strokeOpacity={0.6}
            />
          </g>
        ))}
      </svg>
      
      {/* Processing HUD */}
      <div className="absolute bottom-4 left-6 flex items-center gap-2">
        <Loader2 size={10} className="text-indigo-500 animate-spin" />
        <span className="text-[8px] font-mono text-indigo-400 uppercase tracking-widest opacity-60">Scanning Nodes...</span>
      </div>
    </div>
  );
}

/* ─── Animated SVG visual for RL Autopilot card ─── */
function ParetoVis() {
  const points = [
    [20, 140], [50, 120], [90, 100], [140, 80], [190, 65], [250, 55], [320, 48]
  ];
  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ');

  return (
    <div className="relative w-full h-full">
      <svg viewBox="0 0 360 160" style={{ width: '100%', height: '100%' }}>
        <defs>
          <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#6366f1" />
            <stop offset="100%" stopColor="#818cf8" stopOpacity="0.3" />
          </linearGradient>
          <linearGradient id="areaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Path Area */}
        <motion.path
          d={`${pathD} L${points[points.length-1][0]},155 L${points[0][0]},155 Z`}
          fill="url(#areaGrad)"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        />

        {/* Main Curve */}
        <motion.path
          id="pareto-curve"
          d={pathD}
          fill="none"
          stroke="url(#lineGrad)"
          strokeWidth={2}
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          whileInView={{ pathLength: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 2, ease: "easeInOut" }}
        />

        {/* Traveling Pulse Dot */}
        <motion.circle
          r={3}
          fill="#818cf8"
          style={{ offsetPath: `path('${pathD}')` }}
          animate={{ offsetDistance: ["0%", "100%"] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.circle
          r={6}
          fill="#818cf8"
          fillOpacity={0.2}
          style={{ offsetPath: `path('${pathD}')` }}
          animate={{ offsetDistance: ["0%", "100%"], scale: [1, 1.5, 1] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />

        {/* Target Points */}
        <motion.circle cx={250} cy={55} r={4}
          fill="#818cf8" stroke="rgba(129,140,248,0.3)" strokeWidth={8}
          initial={{ scale: 0 }} whileInView={{ scale: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 1.5, type: "spring" }}
        />
        <motion.circle cx={90} cy={100} r={3}
          fill="#ef4444" stroke="rgba(239,68,68,0.25)" strokeWidth={6}
          initial={{ scale: 0 }} whileInView={{ scale: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 1.2, type: "spring" }}
        />
      </svg>

      <div className="absolute bottom-4 right-6 flex items-center gap-2">
        <div className="flex gap-0.5">
          {[0,1,2].map(i => (
            <motion.div key={i} className="w-0.5 h-2 bg-indigo-500/40"
              animate={{ scaleY: [1, 2, 1] }}
              transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
            />
          ))}
        </div>
        <span className="text-[8px] font-mono text-indigo-400 uppercase tracking-widest opacity-60">Optimizing Policy...</span>
      </div>
    </div>
  );
}

/* ─── Tilt card wrapper with enhanced interactions ─── */
function TiltCard({ children, className = '', style = {} }: {
  children: React.ReactNode; className?: string; style?: React.CSSProperties;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState(false);

  const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = ref.current!;
    const r = el.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width - 0.5) * 6;
    const y = ((e.clientY - r.top) / r.height - 0.5) * -6;
    el.style.transform = `perspective(1000px) rotateX(${y}deg) rotateY(${x}deg) translateY(-8px)`;
    el.style.borderColor = 'rgba(99,102,241,0.3)';
    el.style.boxShadow = `0 30px 60px -15px rgba(0,0,0,0.6), 0 0 40px -10px rgba(99,102,241,0.15)`;
  };

  const onLeave = () => {
    const el = ref.current!;
    el.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0)';
    el.style.borderColor = 'rgba(255,255,255,0.05)';
    el.style.boxShadow = '0 10px 40px -10px rgba(0,0,0,0.5)';
  };

  return (
    <div 
      ref={ref} 
      className={`relative overflow-hidden rounded-[2.5rem] border border-white/5 transition-all duration-500 ease-out ${className}`}
      onMouseMove={onMove} 
      onMouseLeave={onLeave}
      onMouseEnter={() => setHovered(true)}
      onMouseUp={() => setHovered(false)}
    >
      {/* Inner depth gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent pointer-events-none" />
      {children}
    </div>
  );
}

const STATS = [
  { value: '1.8%',    label: 'Final Bias Gap',       sub: 'Reduced from 24%' },
  { value: '83.8%',   label: 'Accuracy Maintained',  sub: 'After autopilot fix' },
  { value: '0.2ms',   label: 'Causal Scan Latency',  sub: 'Real-time detection' },
  { value: '92.4',    label: 'Fairness Score',        sub: 'Out of 100' },
];

export function ModulesSection() {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"]
  });

  const opacity = useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0, 1, 1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.2], [0.95, 1]);

  return (
    <motion.section 
      ref={containerRef}
      style={{ opacity, scale }}
      id="architecture" 
      className="py-40 px-6 max-w-7xl mx-auto relative"
    >
      {/* Section header */}
      <div className="mb-24 relative">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
          className="flex items-center gap-4 mb-6"
        >
          <div className="w-12 h-px bg-indigo-500/50" />
          <span className="text-[0.65rem] font-bold tracking-[0.3em] text-indigo-400 uppercase">System Intelligence</span>
        </motion.div>
        
        <motion.h2
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 1, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          className="text-5xl md:text-7xl font-serif text-white leading-[0.9] tracking-tight max-w-xl"
        >
          The Core <br /> <span className="text-zinc-600">Architecture.</span>
        </motion.h2>
      </div>

      {/* Cards grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-24">
        {/* ── Card 1: Causal Engine ── */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 1.2, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
        >
          <TiltCard className="bg-[#0a0a0f]/80 backdrop-blur-xl h-full flex flex-col group">
            <div className="h-72 p-10 border-b border-white/5 relative bg-gradient-to-br from-indigo-500/5 to-transparent overflow-hidden">
              <CausalGraphVis />
              <div className="absolute top-8 right-8 px-3 py-1 bg-indigo-500/10 border border-indigo-500/20 rounded-full backdrop-blur-md">
                <span className="text-[0.6rem] font-bold tracking-widest text-indigo-300 uppercase font-mono">DAG-GNN Engine</span>
              </div>
              {/* Animated scan line */}
              <motion.div 
                animate={{ top: ['0%', '100%', '0%'] }}
                transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                className="absolute left-0 w-full h-px bg-indigo-500/10 opacity-30 pointer-events-none" 
              />
            </div>
            <div className="p-10 flex flex-col gap-6 flex-1">
              <h3 className="text-3xl font-serif text-white">Causal Discovery</h3>
              <p className="text-zinc-500 leading-relaxed font-light text-lg">
                Moving beyond correlation to understand <span className="text-white">why</span>. Our engine maps non-linear causal chains in real-time, neutralizing proxy discrimination before it reaches inference.
              </p>
              <div className="pt-8 border-t border-white/5 mt-auto flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Search size={14} className="text-indigo-400" />
                  <span className="text-[0.65rem] font-bold tracking-widest text-zinc-500 uppercase">Latency: 0.2ms</span>
                </div>
                <Link href="/causal-graph" className="group/btn flex items-center gap-2 text-white font-bold text-[10px] tracking-[0.2em] uppercase">
                  Observe <ArrowRight size={14} className="group-hover/btn:translate-x-1 transition-transform" />
                </Link>
              </div>
            </div>
          </TiltCard>
        </motion.div>

        {/* ── Card 2: RL Autopilot ── */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 1.2, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
        >
          <TiltCard className="bg-[#0a0a0f]/80 backdrop-blur-xl h-full flex flex-col group">
            <div className="h-72 p-10 border-b border-white/5 relative bg-gradient-to-br from-purple-500/5 to-transparent overflow-hidden">
              <ParetoVis />
              <div className="absolute top-8 right-8 px-3 py-1 bg-purple-500/10 border border-purple-500/20 rounded-full backdrop-blur-md">
                <span className="text-[0.6rem] font-bold tracking-widest text-purple-300 uppercase font-mono">PPO v4.2 Alpha</span>
              </div>
              <motion.div 
                animate={{ top: ['100%', '0%', '100%'] }}
                transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                className="absolute left-0 w-full h-px bg-purple-500/10 opacity-30 pointer-events-none" 
              />
            </div>
            <div className="p-10 flex flex-col gap-6 flex-1">
              <h3 className="text-3xl font-serif text-white">RL Autopilot</h3>
              <p className="text-zinc-500 leading-relaxed font-light text-lg">
                Reinforcement learning that scales. AEGIS navigates the accuracy-fairness Pareto frontier autonomously, correcting bias without degrading model performance.
              </p>
              <div className="pt-8 border-t border-white/5 mt-auto flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Zap size={14} className="text-purple-400" />
                  <span className="text-[0.65rem] font-bold tracking-widest text-zinc-500 uppercase">Protocol: RL-102</span>
                </div>
                <Link href="/autopilot" className="group/btn flex items-center gap-2 text-white font-bold text-[10px] tracking-[0.2em] uppercase">
                  Engage <ArrowRight size={14} className="group-hover/btn:translate-x-1 transition-transform" />
                </Link>
              </div>
            </div>
          </TiltCard>
        </motion.div>
      </div>

      {/* Stats Row */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 1.2, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="grid grid-cols-2 md:grid-cols-4 gap-4 p-10 bg-white/[0.01] border border-white/5 rounded-[3.5rem] backdrop-blur-3xl relative overflow-hidden"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/[0.01] via-transparent to-purple-500/[0.01] pointer-events-none" />
        {STATS.map((s, i) => (
          <div key={i} className="text-center p-6 border-r border-white/5 last:border-0 relative group/stat">
            <p className="text-4xl md:text-5xl font-serif text-white mb-3 leading-none transition-transform group-hover/stat:scale-110 duration-500">{s.value}</p>
            <p className="text-[0.65rem] font-bold tracking-[0.3em] text-zinc-500 uppercase mb-2">{s.label}</p>
            <div className="h-px w-8 bg-zinc-800 mx-auto group-hover/stat:w-16 group-hover/stat:bg-indigo-500/50 transition-all duration-500" />
          </div>
        ))}
      </motion.div>
    </motion.section>
  );
}

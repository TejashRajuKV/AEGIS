'use client';
import { motion } from 'framer-motion';
import { DriftStatusBanner } from '@/components/dashboard/DriftStatusBanner';
import { SystemHealthCard } from '@/components/dashboard/SystemHealthCard';
import { ActiveModelPanel } from '@/components/dashboard/ActiveModelPanel';
import { FairnessScoreGauge } from '@/components/dashboard/FairnessScoreGauge';
import { MetricTrendChart } from '@/components/dashboard/MetricTrendChart';
import { ParetoFrontPlot } from '@/components/dashboard/ParetoFrontPlot';
import { RecentAlertsFeed } from '@/components/dashboard/RecentAlertsFeed';
import { QuickActionsGrid } from '@/components/dashboard/QuickActionsGrid';
import { ToastProvider } from '@/components/ui/Toast';
import { Activity, Shield, LayoutDashboard, Globe, Zap } from 'lucide-react';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.12,
      delayChildren: 0.1
    }
  }
};

const itemVariants = {
  hidden: { y: 30, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { type: 'spring', stiffness: 100, damping: 20 }
  }
};

const floatVariants = {
  animate: {
    y: [0, -10, 0],
    transition: {
      duration: 5,
      repeat: Infinity,
      ease: "easeInOut"
    }
  }
};

export default function DashboardPage() {
  return (
    <ToastProvider>
      <div className="min-h-screen relative overflow-hidden bg-[#050508]">
        {/* Advanced Layered Background */}
        <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))] opacity-[0.05]" />
        
        {/* Signature Drifting Particles */}
        <div className="absolute inset-0 pointer-events-none">
          {[...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              initial={{ x: Math.random() * 100 + '%', y: '110%', opacity: 0 }}
              animate={{ 
                y: '-10%', 
                opacity: [0, 0.4, 0],
                x: (Math.random() * 100) + (Math.sin(i) * 5) + '%'
              }}
              transition={{ 
                duration: 15 + Math.random() * 10, 
                repeat: Infinity, 
                ease: "linear",
                delay: Math.random() * 20
              }}
              className="absolute w-1 h-1 bg-indigo-500/30 rounded-full blur-[1px]"
            />
          ))}
        </div>

        {/* Ambient Glows */}
        <div className="absolute top-[-20%] left-[-10%] w-[800px] h-[800px] bg-indigo-600/10 blur-[150px] rounded-full mix-blend-screen animate-pulse" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-purple-600/10 blur-[120px] rounded-full mix-blend-screen" />

        <motion.div 
          className="page-container relative z-10 py-16"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Enhanced Header */}
          <header className="mb-16">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-8">
              <motion.div variants={itemVariants} className="max-w-3xl">
                <div className="flex items-center gap-3 mb-6">
                  <div className="px-3 py-1 bg-indigo-500/10 border border-indigo-500/20 rounded-full flex items-center gap-2">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
                    </span>
                    <span className="text-[10px] font-mono font-bold tracking-[0.2em] text-indigo-300 uppercase">
                      Live Governance Active
                    </span>
                  </div>
                </div>
                
                <h1 className="text-6xl md:text-8xl font-serif mb-6 bg-gradient-to-r from-white via-white/90 to-white/30 bg-clip-text text-transparent leading-[0.85] tracking-tighter">
                  System <br /> <span className="text-zinc-700">Governance.</span>
                </h1>
                <p className="text-xl text-zinc-500 max-w-xl leading-relaxed font-light">
                  Neutralizing algorithmic bias at the speed of inference. AEGIS is now monitoring <span className="text-zinc-200">2.4M transactions/hr</span>.
                </p>
              </motion.div>
              
              <motion.div 
                variants={itemVariants}
                className="flex flex-col items-end gap-4"
              >
                <div className="flex items-center gap-4 px-6 py-3 bg-white/[0.02] backdrop-blur-2xl border border-white/5 rounded-2xl shadow-2xl">
                  <div className="flex flex-col items-end">
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                      <span className="text-[10px] font-mono text-emerald-400 font-bold uppercase tracking-widest">Global Uplink Stable</span>
                    </div>
                    <span className="text-[9px] font-mono text-zinc-600 mt-0.5">LATENCY: 0.2ms • PROTOCOL: QUIC-V3</span>
                  </div>
                  <div className="w-px h-10 bg-white/5" />
                  <div className="p-2 bg-indigo-500/10 rounded-xl border border-indigo-500/20">
                    <Globe size={18} className="text-indigo-400" />
                  </div>
                </div>
              </motion.div>
            </div>
          </header>

          {/* Global Alert Banner */}
          <motion.div variants={itemVariants} className="mb-10">
            <DriftStatusBanner />
          </motion.div>

          {/* Main Dashboard Grid */}
          <div className="grid grid-cols-12 gap-8">
            
            {/* Left Column: Health & Model Status */}
            <div className="col-span-12 lg:col-span-8 grid grid-cols-1 md:grid-cols-2 gap-8">
              <motion.div variants={itemVariants} whileHover={{ y: -5 }} className="h-full">
                <SystemHealthCard />
              </motion.div>
              <motion.div variants={itemVariants} whileHover={{ y: -5 }} className="h-full">
                <ActiveModelPanel />
              </motion.div>
              <motion.div variants={itemVariants} className="col-span-1 md:col-span-2">
                <MetricTrendChart />
              </motion.div>
            </div>

            {/* Right Column: Fairness & Quick Actions */}
            <div className="col-span-12 lg:col-span-4 flex flex-col gap-8">
              <motion.div variants={itemVariants} animate="animate" variants={{...itemVariants, ...floatVariants}}>
                <FairnessScoreGauge />
              </motion.div>
              <motion.div variants={itemVariants} className="flex-1">
                <RecentAlertsFeed />
              </motion.div>
            </div>

            {/* Bottom Row: Charts & Actions */}
            <motion.div variants={itemVariants} className="col-span-12 lg:col-span-7">
              <ParetoFrontPlot />
            </motion.div>
            
            <motion.div variants={itemVariants} className="col-span-12 lg:col-span-5">
              <QuickActionsGrid />
            </motion.div>

          </div>
        </motion.div>
      </div>
    </ToastProvider>
  );
}



'use client';
import { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle, ArrowRight, Zap } from 'lucide-react';
import { driftService, DriftAlertItem } from '@/services';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';

export function DriftStatusBanner() {
  const [alerts, setAlerts] = useState<DriftAlertItem[]>([]);
  
  useEffect(() => {
    async function load() {
      try {
        const res = await driftService.getAlerts();
        setAlerts(res.alerts.filter(a => !a.resolved));
      } catch (e) {
        console.error(e);
      }
    }
    load();
    const id = setInterval(load, 10000);
    return () => clearInterval(id);
  }, []);

  const critical = alerts.filter(a => a.severity === 'critical').length;
  const warning = alerts.filter(a => a.severity === 'warning').length;

  if (alerts.length === 0) {
    return (
      <motion.div 
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative group overflow-hidden rounded-2xl border border-emerald-500/10 bg-emerald-500/[0.02] p-4 flex items-center justify-between gap-6 backdrop-blur-md shadow-[0_10px_30px_rgba(0,0,0,0.2)]"
      >
        <div className="flex items-center gap-4">
          <div className="relative flex items-center justify-center">
            <motion.div 
              animate={{ scale: [1, 1.2, 1], opacity: [0.2, 0.4, 0.2] }}
              transition={{ duration: 3, repeat: Infinity }}
              className="absolute w-8 h-8 bg-emerald-500 rounded-full blur-lg"
            />
            <CheckCircle size={20} className="text-emerald-400 relative z-10" />
          </div>
          <div>
            <h4 className="text-sm font-bold text-emerald-400 tracking-tight">No Drift Detected</h4>
            <p className="text-[11px] text-zinc-500 font-medium">All monitored features are within expected distribution limits.</p>
          </div>
        </div>
        
        <div className="hidden md:flex items-center gap-3 px-3 py-1 bg-white/[0.03] border border-white/5 rounded-full">
          <Zap size={10} className="text-emerald-500" />
          <span className="text-[9px] font-mono text-zinc-500 uppercase tracking-widest">Latency: 0.1ms</span>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative group overflow-hidden rounded-2xl border border-rose-500/20 bg-rose-500/[0.03] p-4 flex items-center justify-between gap-6 backdrop-blur-md shadow-[0_10px_40px_rgba(244,63,94,0.1)]"
    >
      <motion.div 
        animate={{ opacity: [0.05, 0.1, 0.05] }}
        transition={{ duration: 2, repeat: Infinity }}
        className="absolute inset-0 bg-rose-500 pointer-events-none"
      />
      
      <div className="flex items-center gap-4 relative z-10">
        <div className="relative flex items-center justify-center">
          <motion.div 
            animate={{ scale: [1, 1.4, 1], opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="absolute w-10 h-10 bg-rose-500 rounded-full blur-xl"
          />
          <AlertTriangle size={24} className="text-rose-400 relative z-10" />
        </div>
        <div>
          <h4 className="text-[0.95rem] font-bold text-rose-400 tracking-tight mb-0.5">Critical Data Drift Alert</h4>
          <p className="text-[11px] text-zinc-400 leading-none">
            <span className="text-rose-300 font-bold">{critical} critical</span> and {warning} warning feature distributions have shifted.
          </p>
        </div>
      </div>

      <Link 
        href="/drift-monitor" 
        className="relative z-10 flex items-center gap-3 px-6 py-2.5 bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 rounded-xl text-[10px] font-bold text-rose-300 uppercase tracking-widest transition-all hover:-translate-x-1"
      >
        View Analysis
        <ArrowRight size={14} />
      </Link>
    </motion.div>
  );
}


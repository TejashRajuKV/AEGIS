'use client';
import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Skeleton } from '@/components/ui/Skeleton';
import { pingBackend, apiGet } from '@/services';
import { Server, Activity, Database, Cpu, Zap, Globe } from 'lucide-react';
import dynamic from 'next/dynamic';

const GlobalMonitor = dynamic(() => import('./GlobalMonitor').then(mod => mod.GlobalMonitor), {
  ssr: false,
  loading: () => <div className="w-full h-full bg-indigo-500/5 animate-pulse rounded-full" />
});

export function SystemHealthCard() {
  const [health, setHealth] = useState<{ status: string; version: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [uptime, setUptime] = useState(0);

  useEffect(() => {
    async function checkHealth() {
      try {
        const isUp = await pingBackend();
        if (isUp) {
          const res = await apiGet<{ status: string; version: string }>('/api/health');
          setHealth(res);
        } else {
          setHealth({ status: 'offline', version: 'N/A' });
        }
      } catch {
        setHealth({ status: 'offline', version: 'N/A' });
      } finally {
        setLoading(false);
      }
    }
    checkHealth();
    const id = setInterval(checkHealth, 30_000);
    const timer = setInterval(() => setUptime(prev => prev + 1), 1000);
    return () => { clearInterval(id); clearInterval(timer); };
  }, []);

  if (loading) {
    return (
      <Card className="h-full bg-white/[0.01] border-white/5">
        <CardHeader><CardTitle>System Health</CardTitle></CardHeader>
        <div className="grid grid-cols-2 gap-4 mt-2 p-6">
          {[1, 2, 3, 4].map(i => <Skeleton key={i} height={60} className="rounded-xl bg-white/5" />)}
        </div>
      </Card>
    );
  }

  const isHealthy = health?.status === 'ok' || health?.status === 'healthy';

  const stats = [
    { label: 'Gateway', value: isHealthy ? 'ONLINE' : 'DOWN', icon: Server, color: isHealthy ? 'text-emerald-400' : 'text-rose-400' },
    { label: 'Latency', value: '0.2ms', icon: Zap, color: 'text-indigo-400' },
    { label: 'Uptime', value: `${Math.floor(uptime / 60)}m ${uptime % 60}s`, icon: Activity, color: 'text-sky-400' },
    { label: 'Engine', value: 'ACTIVE', icon: Cpu, color: 'text-amber-400' },
  ];

  return (
    <Card className="h-full relative overflow-hidden group bg-white/[0.01] border-white/5 shadow-2xl">
      {/* Background Globe Visual */}
      <div className="absolute right-[-20%] top-[-10%] w-[120%] h-[120%] opacity-20 pointer-events-none group-hover:opacity-40 transition-opacity duration-1000">
        <GlobalMonitor />
      </div>

      <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/[0.05] to-transparent pointer-events-none" />
      
      <CardHeader className="flex flex-row justify-between items-start border-none relative z-10 p-8 pb-4">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 mb-1">
            <div className={`w-1.5 h-1.5 rounded-full ${isHealthy ? 'bg-emerald-500' : 'bg-rose-500'} shadow-[0_0_10px_currentColor] animate-pulse`} />
            <span className="text-[10px] font-mono font-bold text-indigo-400 uppercase tracking-[0.3em]">Neural Network Health</span>
          </div>
          <CardTitle size="xl" className="text-3xl font-serif font-bold text-white tracking-tight">System Core.</CardTitle>
          <p className="text-xs text-zinc-500 max-w-[200px] leading-relaxed">Monitoring 142 globally distributed inference nodes.</p>
        </div>
        <Badge className={`${isHealthy ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border-rose-500/20'} text-[9px] tracking-widest uppercase font-bold px-3 py-1 rounded-full`}>
          {isHealthy ? 'Operational' : 'Critical'}
        </Badge>
      </CardHeader>
      
      <div className="grid grid-cols-2 gap-4 p-8 pt-4 relative z-10">
        {stats.map((stat, i) => (
          <motion.div 
            key={stat.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="flex flex-col gap-2 p-4 bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-2xl hover:bg-white/[0.05] transition-all duration-300"
          >
            <div className="flex items-center gap-2">
              <stat.icon size={12} className={stat.color} />
              <span className="text-[9px] uppercase tracking-widest text-zinc-500 font-bold">{stat.label}</span>
            </div>
            <p className="text-lg font-mono font-bold text-white tracking-tighter">{stat.value}</p>
          </motion.div>
        ))}
      </div>

      {/* Decorative HUD Elements */}
      <div className="absolute bottom-4 left-8 right-8 h-px bg-white/5" />
      <div className="absolute bottom-2 left-8 flex gap-4 text-[8px] font-mono text-zinc-700">
        <span>X: 104.22.11.4</span>
        <span>Y: 92.112.0.1</span>
        <span>Z: QUIC-V3</span>
      </div>
    </Card>
  );
}



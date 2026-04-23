'use client';
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis } from 'recharts';
import { autopilotService, ParetoPoint } from '@/services';
import { Target, Info } from 'lucide-react';

export function ParetoFrontPlot() {
  const [data, setData] = useState<ParetoPoint[]>([]);

  useEffect(() => {
    async function load() {
      try {
        const res = await autopilotService.getParetoFrontier();
        if (res.points?.length) setData(res.points);
      } catch (e) {
        console.error('Failed to load pareto front:', e);
      }
    }
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <Card className="h-[400px] flex flex-col relative overflow-hidden group">
      {/* Background decoration */}
      <div className="absolute top-[-20px] right-[-20px] opacity-[0.02] rotate-12">
        <Target size={160} />
      </div>

      <CardHeader className="flex flex-row justify-between items-center border-none pb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-500/10 rounded-lg">
            <Target size={18} className="text-indigo-400" />
          </div>
          <CardTitle size="lg">Optimization Frontier</CardTitle>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 bg-white/[0.03] border border-white/5 rounded-full">
          <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-[0.1em]">RL Agent Trace</span>
        </div>
      </CardHeader>

      <div className="flex-1 min-h-0 pr-4">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.03)" />
            <XAxis 
              type="number" 
              dataKey="fairness" 
              name="Fairness" 
              domain={[60, 100]} 
              axisLine={false} 
              tickLine={false} 
              tick={{ fontSize: 10, fill: '#71717a' }} 
              tickFormatter={v => `${v}%`}
              label={{ value: 'FAIRNESS', position: 'bottom', offset: 0, fontSize: 9, fill: '#52525b', letterSpacing: '0.1em' }}
            />
            <YAxis 
              type="number" 
              dataKey="accuracy" 
              name="Accuracy" 
              domain={[70, 96]} 
              axisLine={false} 
              tickLine={false} 
              tick={{ fontSize: 10, fill: '#71717a' }} 
              tickFormatter={v => `${v}%`}
              label={{ value: 'ACCURACY', angle: -90, position: 'insideLeft', offset: 10, fontSize: 9, fill: '#52525b', letterSpacing: '0.1em' }}
            />
            <ZAxis type="number" range={[100, 100]} />
            <Tooltip
              cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1, strokeDasharray: '4 4' }}
              contentStyle={{ 
                background: 'rgba(15, 15, 20, 0.95)', 
                border: '1px solid rgba(255,255,255,0.1)', 
                borderRadius: '12px',
                backdropFilter: 'blur(12px)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
                padding: '12px'
              }}
              itemStyle={{ color: '#fff', fontSize: '13px', fontWeight: 600, padding: '2px 0' }}
              labelStyle={{ display: 'none' }}
              formatter={(v: number, name: string) => [`${v.toFixed(1)}%`, name]}
            />
            <Scatter 
              name="Trade-off" 
              data={data} 
              fill="#818cf8" 
              className="drop-shadow-[0_0_8px_rgba(129,140,248,0.5)]"
              stroke="rgba(255,255,255,0.2)"
              strokeWidth={1}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="p-4 bg-white/[0.02] border-t border-white/5 flex items-start gap-3 mt-4">
        <Info size={14} className="text-indigo-400 mt-0.5 shrink-0" />
        <p className="text-[10px] text-zinc-500 leading-normal">
          Each point represents a unique model configuration found by the RL Optimizer. 
          The curve illustrates the <span className="text-zinc-300">Pareto Optimal Frontier</span> where fairness cannot be increased without sacrificing accuracy.
        </p>
      </div>
    </Card>
  );
}


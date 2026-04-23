'use client';
import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, AreaChart, Area } from 'recharts';
import { fairnessService, TrendPoint } from '@/services';
import { TrendingUp } from 'lucide-react';

export function MetricTrendChart() {
  const [data, setData] = useState<TrendPoint[]>([]);

  useEffect(() => {
    async function load() {
      try {
        const res = await fairnessService.getDashboardSummary();
        setData(res.trend);
      } catch (e) {
        console.error('Failed to load trend data', e);
      }
    }
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <Card className="h-[400px] flex flex-col group">
      <CardHeader className="flex flex-row justify-between items-center border-none pb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/10 rounded-lg">
            <TrendingUp size={18} className="text-emerald-400" />
          </div>
          <CardTitle size="lg">Fairness vs. Accuracy</CardTitle>
        </div>
        <div className="flex gap-4 px-3 py-1.5 bg-white/[0.03] border border-white/5 rounded-full">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-indigo-500" />
            <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-wider">Fairness</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-wider">Accuracy</span>
          </div>
        </div>
      </CardHeader>
      
      <div className="flex-1 min-h-0 pr-4">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id="colorFairness" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.1}/>
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.1}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.03)" />
            <XAxis 
              dataKey="day" 
              axisLine={false} 
              tickLine={false} 
              tick={{ fontSize: 10, fill: '#71717a' }} 
              dy={10}
            />
            <YAxis 
              domain={[75, 100]} 
              axisLine={false} 
              tickLine={false} 
              tick={{ fontSize: 10, fill: '#71717a' }} 
              tickFormatter={v => `${v}%`} 
            />
            <Tooltip
              contentStyle={{ 
                background: 'rgba(15, 15, 20, 0.9)', 
                border: '1px solid rgba(255,255,255,0.1)', 
                borderRadius: '12px',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
              }}
              itemStyle={{ fontSize: '12px', fontWeight: 600, color: '#fff' }}
              labelStyle={{ fontSize: '11px', color: '#71717a', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}
              formatter={(v: number) => [`${v.toFixed(1)}%`]}
            />
            <Area 
              type="monotone" 
              dataKey="fairness" 
              stroke="#6366f1" 
              strokeWidth={3} 
              fillOpacity={1} 
              fill="url(#colorFairness)" 
              animationDuration={1500}
            />
            <Area 
              type="monotone" 
              dataKey="accuracy" 
              stroke="#10b981" 
              strokeWidth={3} 
              fillOpacity={1} 
              fill="url(#colorAccuracy)" 
              animationDuration={1500}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}


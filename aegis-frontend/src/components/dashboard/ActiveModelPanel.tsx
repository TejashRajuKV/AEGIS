'use client';
import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { modelService, ModelInfo, TrainModelResponse } from '@/services';
import { Play, Boxes, CheckCircle2, AlertCircle, Cpu, ShieldCheck, Box, ChevronRight, Activity } from 'lucide-react';
import { useToast } from '@/components/ui/Toast';

export function ActiveModelPanel() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('logistic_regression');
  const [isTraining, setIsTraining] = useState(false);
  const [lastTrainResult, setLastTrainResult] = useState<TrainModelResponse | null>(null);
  const [mounted, setMounted] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    async function load() {
      try {
        const res = await modelService.listModels();
        
        const realModels = res.models || [];
        
        const validatedModels = realModels.map(m => {
          const isUuid = m.name?.includes('-') && m.name?.length > 20;
          const displayName = (!m.name || isUuid) 
            ? (m.model_type || 'Unknown Model')
            : m.name;
          
          return {
            ...m,
            name: displayName.replace(/_/g, ' ')
          };
        });

        setModels(validatedModels);
        const active = validatedModels.find(m => m.is_active);
        if (active) setSelectedModel(active.name);
        else if (validatedModels.length > 0) setSelectedModel(validatedModels[0].name);
      } catch (e) {
        console.error('Failed to load models:', e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const handleRetrain = async () => {
    if (!selectedModel) return;
    setIsTraining(true);
    try {
      const res = await modelService.trainModel(selectedModel, { dataset_name: 'compas', retrain: true });
      setLastTrainResult(res);
      toast('success', 'Model Optimized', `Fairness improved to ${(res.test_accuracy * 100).toFixed(1)}%`);
    } catch (e: any) {
      toast('error', 'Optimization Failed', e.message);
    } finally {
      setIsTraining(false);
    }
  };

  const active = models.find(m => m.name === selectedModel);

  return (
    <Card className="h-full flex flex-col relative overflow-hidden group hover:shadow-[0_25px_50px_rgba(0,0,0,0.5)] transition-all duration-700 bg-white/[0.01] border-white/5 shadow-inner">
      <div className="absolute inset-0 bg-gradient-to-br from-purple-500/[0.02] to-transparent pointer-events-none" />
      
      <CardHeader className="flex flex-row justify-between items-center border-none relative z-10 pb-4">
        <div className="flex flex-col gap-1">
          <CardTitle size="lg" className="text-zinc-200">Model Configuration</CardTitle>
          <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">Pipeline: Stage-04-A</span>
        </div>
        <div className="p-2 bg-purple-500/10 rounded-xl border border-purple-500/20 shadow-inner">
          <Cpu size={14} className="text-purple-400" />
        </div>
      </CardHeader>

      <div className="flex-1 p-6 pt-0 relative z-10 flex flex-col gap-6">
        <div className="relative group/select">
          <Select
            options={models.map(m => ({ label: (m.name || 'Unknown').replace(/_/g, ' '), value: m.name || '' }))}
            value={selectedModel}
            onChange={setSelectedModel}
            className="bg-white/[0.02] border-white/10 rounded-2xl h-12"
          />
        </div>

        <AnimatePresence mode="wait">
          {active && (
            <motion.div 
              key={active.model_id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="flex items-center gap-4 p-4 bg-white/[0.03] border border-white/5 rounded-[2rem] shadow-inner"
            >
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-500/20 to-indigo-500/20 flex items-center justify-center border border-white/10 shadow-lg group-hover:scale-105 transition-transform duration-500">
                <Box size={24} className="text-white opacity-80" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-lg font-serif font-bold text-white mb-0.5 truncate">{(active.name || 'Unknown').replace(/_/g, ' ')}</div>
                <div className="flex items-center gap-2">
                  <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[9px] px-1.5 h-4 font-bold tracking-tighter">STABLE</Badge>
                  <span className="text-[10px] font-mono text-zinc-500 tracking-widest uppercase">v{active.version}</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid grid-cols-2 gap-4">
          {[
            { label: 'Architecture', value: active?.model_type || 'Unknown', icon: Boxes, color: 'text-zinc-500' },
            { label: 'Governance', value: 'Enabled', icon: ShieldCheck, color: 'text-emerald-400' }
          ].map((s, i) => (
            <div key={i} className="p-4 bg-white/[0.02] border border-white/5 rounded-2xl flex flex-col gap-2 shadow-inner group/stat hover:bg-white/[0.04] transition-colors">
              <div className="flex items-center gap-2 text-zinc-500">
                <s.icon size={12} className={`${s.color} opacity-70 group-hover/stat:opacity-100 transition-opacity`} />
                <span className="text-[9px] uppercase tracking-[0.2em] font-bold">{s.label}</span>
              </div>
              <span className="text-xs font-medium text-zinc-300 pl-5 tracking-tight">{s.value}</span>
            </div>
          ))}
        </div>

        <div className="mt-auto flex flex-col gap-3">
          <Button
            variant="primary"
            onClick={handleRetrain}
            loading={isTraining}
            leftIcon={<Activity size={14} className="animate-pulse" />}
            className="w-full justify-center py-4 h-auto text-[10px] font-bold tracking-[0.3em] uppercase bg-indigo-600 hover:bg-indigo-500 shadow-[0_10px_20px_rgba(79,70,229,0.2)] rounded-2xl"
          >
            {isTraining ? 'Executing Optimization...' : 'Optimize Pipeline'}
          </Button>
          <div className="flex items-center justify-center gap-2 text-[8px] font-mono text-zinc-600 tracking-widest uppercase">
            <span className="w-1 h-1 rounded-full bg-zinc-700" />
            Last Check: {mounted ? new Date().toLocaleTimeString() : '--:--:--'}
            <span className="w-1 h-1 rounded-full bg-zinc-700" />
          </div>
        </div>
      </div>
    </Card>
  );
}



'use client';

import { useEffect, useRef, useState } from 'react';

const LINES = [
  { delay: 0,    text: 'AEGIS.TERMINAL_ SYSTEM. AUTH_CODE.0X_209', type:'meta' },
  { delay: 400,  text: '', type: 'meta' },
  { delay: 900,  text: '[INIT] Initializing Causal Engine v2.5...', type:'info' },
  { delay: 1600, text: '[OK] Connection established with Model_Cluster_7', type:'ok' },
  { delay: 2200, text: '[LOG] Scanning parameter space for bias vectors...', type:'log' },
  { delay: 2900, text: '[WARN] Potential drift detected in Sub-cluster 14B (0.0042)', type:'warn' },
  { delay: 3500, text: '[ACTION] Applying RL-correction patch #462...', type:'action' },
  { delay: 4200, text: '[SUCCESS] Parity restored. Health: 99.98%', type:'success' },
  { delay: 4800, text: '[LOG] ----------------------------------------', type:'log' },
  { delay: 5400, text: '[SYS] Waiting for next governance cycle...', type:'info' },
  { delay: 6000, text: '_', type:'cursor' },
];

const color: Record<string,string> = {
  meta:   '#52525b',
  info:   '#60a5fa', // Blue
  ok:     '#e4e4e7', // White-ish
  log:    '#71717a', // Muted grey
  warn:   '#facc15', // Yellow
  action: '#818cf8', // Indigo
  success:'#34d399', // Green
  cursor: '#e4e4e7',
};

export function TerminalWindow() {
  const [visible, setVisible] = useState<typeof LINES>([]);
  const [started, setStarted] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting && !started) setStarted(true); }, { threshold: 0.3 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, [started]);

  useEffect(() => {
    if (!started) return;
    const timers = LINES.map(l => setTimeout(() => {
      setVisible(v => [...v, l]);
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, l.delay));
    return () => timers.forEach(clearTimeout);
  }, [started]);

  return (
    <div ref={ref} className="glass-panel" style={{ padding: '1rem', background: 'rgba(10,10,12,0.8)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: 12, overflow: 'hidden', fontFamily: 'var(--font-fira-code), monospace' }}>
      {/* Fake window controls */}
      <div style={{ display: 'flex', gap: 6, marginBottom: '1.5rem', padding: '0.5rem 0.5rem 0' }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444', opacity: 0.5 }} />
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#facc15', opacity: 0.5 }} />
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#22c55e', opacity: 0.5 }} />
      </div>

      {/* Output */}
      <div style={{ padding: '0 0.5rem 1rem', minHeight: 280, maxHeight: 320, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {visible.map((l, i) => (
          <p key={i} style={{ fontSize: '0.75rem', lineHeight: 1.6, color: color[l.type] || '#a1a1aa',
            animation: 'fadeIn 0.2s ease forwards',
            fontWeight: 500, letterSpacing: '0.05em'
          }}>
            {l.type === 'cursor'
              ? <span style={{ animation: 'blinkCursor 1s step-end infinite', background: '#e4e4e7', display: 'inline-block', width: 6, height: 12 }} />
              : l.text
            }
          </p>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

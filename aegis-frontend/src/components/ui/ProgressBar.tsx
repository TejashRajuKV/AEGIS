'use client';
import React from 'react';

interface ProgressBarProps {
  value: number;    // 0–100
  max?: number;
  label?: string;
  showValue?: boolean;
  color?: string;
  height?: number;
  style?: React.CSSProperties;
}

export function ProgressBar({ value, max = 100, label, showValue, color = 'var(--accent)', height = 6, style }: ProgressBarProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div style={{ width: '100%', ...style }}>
      {(label || showValue) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
          {label && <span style={{ fontSize: '0.78rem', color: '#a1a1aa', fontWeight: 500 }}>{label}</span>}
          {showValue && <span style={{ fontSize: '0.78rem', color: '#71717a', fontFamily: 'var(--font-mono)' }}>{value.toFixed(1)}%</span>}
        </div>
      )}
      <div className="progress-track" style={{ height }}>
        <div
          className="progress-fill"
          style={{ width: `${pct}%`, background: color, animation: 'none', boxShadow: 'none' }}
        />
      </div>
    </div>
  );
}

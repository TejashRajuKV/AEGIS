'use client';
import React, { useState } from 'react';

interface SliderProps {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  showValue?: boolean;
  formatValue?: (v: number) => string;
  style?: React.CSSProperties;
}

export function Slider({ value, onChange, min = 0, max = 100, step = 1, label, showValue, formatValue, style }: SliderProps) {
  const display = formatValue ? formatValue(value) : String(value);
  return (
    <div style={{ width: '100%', ...style }}>
      {(label || showValue) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
          {label && <span style={{ fontSize: '0.78rem', color: '#a1a1aa', fontWeight: 500 }}>{label}</span>}
          {showValue && <span style={{ fontSize: '0.78rem', color: 'var(--accent-light)', fontFamily: 'var(--font-mono)' }}>{display}</span>}
        </div>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{
          width: '100%',
          appearance: 'none',
          height: 4,
          background: `linear-gradient(to right, #6366f1 ${((value - min) / (max - min)) * 100}%, rgba(255,255,255,0.1) ${((value - min) / (max - min)) * 100}%)`,
          borderRadius: 100,
          outline: 'none',
          cursor: 'pointer',
        }}
      />
    </div>
  );
}

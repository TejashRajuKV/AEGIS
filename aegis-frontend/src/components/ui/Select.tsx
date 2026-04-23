'use client';
import React from 'react';
import { ChevronDown } from 'lucide-react';

interface SelectOption { value: string; label: string; }

interface SelectProps {
  label?: string;
  options: SelectOption[];
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  style?: React.CSSProperties;
  className?: string;
}

export function Select({ label, options, value, onChange, placeholder, disabled, style, className = '' }: SelectProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem', width: '100%' }}>
      {label && (
        <label style={{ fontSize: '0.8rem', fontWeight: 600, color: '#a1a1aa', letterSpacing: '0.03em' }}>
          {label}
        </label>
      )}
      <div style={{ position: 'relative' }}>
        <select
          className={`select-field ${className}`}
          value={value}
          disabled={disabled}
          onChange={e => onChange?.(e.target.value)}
          style={style}
        >
          {placeholder && <option value="" disabled className="bg-zinc-900 text-zinc-400">{placeholder}</option>}
          {options.map(o => (
            <option key={o.value} value={o.value} className="bg-zinc-900 text-white">
              {o.label}
            </option>
          ))}
        </select>
        <ChevronDown
          size={14}
          style={{
            position: 'absolute', right: '1rem', top: '50%',
            transform: 'translateY(-50%)', color: '#71717a', pointerEvents: 'none',
          }}
        />
      </div>
    </div>
  );
}

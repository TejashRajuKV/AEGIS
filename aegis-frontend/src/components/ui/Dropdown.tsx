'use client';
import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';

interface DropdownItem { label: string; value: string; icon?: React.ReactNode; danger?: boolean; }

interface DropdownProps {
  trigger: React.ReactNode;
  items: DropdownItem[];
  onSelect?: (value: string) => void;
  align?: 'left' | 'right';
}

export function Dropdown({ trigger, items, onSelect, align = 'left' }: DropdownProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <div ref={ref} style={{ position: 'relative', display: 'inline-block' }}>
      <div onClick={() => setOpen(o => !o)} style={{ cursor: 'pointer' }}>{trigger}</div>
      {open && (
        <div style={{
          position: 'absolute',
          top: 'calc(100% + 6px)',
          [align]: 0,
          minWidth: 180,
          background: 'rgba(16,16,22,0.97)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 'var(--radius-md)',
          boxShadow: 'var(--shadow-lg)',
          padding: '0.375rem',
          zIndex: 150,
          animation: 'slideUp 0.15s ease',
        }}>
          {items.map(item => (
            <button key={item.value}
              onClick={() => { onSelect?.(item.value); setOpen(false); }}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '0.5rem 0.75rem',
                background: 'transparent',
                border: 'none',
                borderRadius: 6,
                color: item.danger ? '#f87171' : '#d4d4d8',
                fontSize: '0.825rem',
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = item.danger ? 'rgba(239,68,68,0.1)' : 'rgba(255,255,255,0.05)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

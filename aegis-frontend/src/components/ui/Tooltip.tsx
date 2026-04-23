'use client';
import React, { useState } from 'react';

interface TooltipProps { children: React.ReactNode; content: string; side?: 'top' | 'bottom' | 'left' | 'right'; }
export function Tooltip({ children, content, side = 'top' }: TooltipProps) {
  const [show, setShow] = useState(false);
  const offsets: Record<string, React.CSSProperties> = {
    top:    { bottom: '100%', left: '50%', transform: 'translateX(-50%)', marginBottom: 8 },
    bottom: { top: '100%', left: '50%', transform: 'translateX(-50%)', marginTop: 8 },
    left:   { right: '100%', top: '50%', transform: 'translateY(-50%)', marginRight: 8 },
    right:  { left: '100%', top: '50%', transform: 'translateY(-50%)', marginLeft: 8 },
  };
  return (
    <div
      style={{ position: 'relative', display: 'inline-flex' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div style={{
          position: 'absolute',
          ...offsets[side],
          background: 'rgba(22,22,30,0.95)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 6,
          padding: '4px 10px',
          fontSize: '0.72rem',
          color: '#d4d4d8',
          whiteSpace: 'nowrap',
          zIndex: 300,
          pointerEvents: 'none',
          animation: 'fadeIn 0.15s ease',
          boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
        }}>
          {content}
        </div>
      )}
    </div>
  );
}

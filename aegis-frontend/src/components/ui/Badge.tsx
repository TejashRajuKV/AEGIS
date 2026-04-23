'use client';
import React from 'react';

type BadgeVariant = 'success' | 'warning' | 'danger' | 'info' | 'neutral';

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  dot?: boolean;
  style?: React.CSSProperties;
}

export function Badge({ variant = 'neutral', children, dot, style }: BadgeProps) {
  const dotColor: Record<BadgeVariant, string> = {
    success: '#4ade80', warning: '#fbbf24', danger: '#f87171', info: '#818cf8', neutral: '#71717a',
  };
  return (
    <span className={`badge badge-${variant}`} style={style}>
      {dot && (
        <span style={{
          display: 'inline-block', width: 5, height: 5, borderRadius: '50%',
          background: dotColor[variant], flexShrink: 0,
        }} />
      )}
      {children}
    </span>
  );
}

'use client';
import React from 'react';
import { CheckCircle, AlertTriangle, XCircle, Info, X } from 'lucide-react';

type AlertVariant = 'success' | 'warning' | 'danger' | 'info';

const icons = { success: CheckCircle, warning: AlertTriangle, danger: XCircle, info: Info };

interface AlertProps {
  variant: AlertVariant;
  title?: string;
  children: React.ReactNode;
  onDismiss?: () => void;
  style?: React.CSSProperties;
}

export function Alert({ variant, title, children, onDismiss, style }: AlertProps) {
  const Icon = icons[variant];
  return (
    <div className={`alert alert-${variant}`} style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start', ...style }}>
      <Icon size={16} style={{ flexShrink: 0, marginTop: 1 }} />
      <div style={{ flex: 1 }}>
        {title && <p style={{ fontWeight: 600, marginBottom: '0.25rem', fontSize: '0.875rem' }}>{title}</p>}
        <div style={{ fontSize: '0.825rem', opacity: 0.9 }}>{children}</div>
      </div>
      {onDismiss && (
        <button onClick={onDismiss} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', opacity: 0.6, padding: 2 }}>
          <X size={14} />
        </button>
      )}
    </div>
  );
}

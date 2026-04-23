'use client';
import React, { createContext, useContext, useState, useCallback } from 'react';
import { X, CheckCircle, AlertTriangle, XCircle, Info } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

interface ToastMessage {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
}

interface ToastContextType {
  toast: (type: ToastType, title: string, message?: string) => void;
}

const ToastContext = createContext<ToastContextType | null>(null);

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) throw new Error('useToast must be used within ToastProvider');
  return context;
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const toast = useCallback((type: ToastType, title: string, message?: string) => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, type, title, message }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 5000);
  }, []);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div style={{
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        zIndex: 9999,
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
        pointerEvents: 'none'
      }}>
        {toasts.map((t) => (
          <ToastItem key={t.id} toast={t} onRemove={() => removeToast(t.id)} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}

function ToastItem({ toast, onRemove }: { toast: ToastMessage, onRemove: () => void }) {
  const icons = { success: CheckCircle, error: XCircle, warning: AlertTriangle, info: Info };
  const colors = { success: '#4ade80', error: '#f87171', warning: '#fbbf24', info: '#818cf8' };
  const Icon = icons[toast.type];

  return (
    <div style={{
      background: 'rgba(22, 22, 30, 0.95)',
      backdropFilter: 'blur(16px)',
      border: `1px solid rgba(255, 255, 255, 0.1)`,
      borderLeft: `4px solid ${colors[toast.type]}`,
      borderRadius: 'var(--radius-md)',
      boxShadow: 'var(--shadow-lg)',
      padding: '16px',
      width: '320px',
      display: 'flex',
      alignItems: 'flex-start',
      gap: '12px',
      animation: 'slideIn 0.3s ease-out',
      pointerEvents: 'auto'
    }}>
      <Icon color={colors[toast.type]} size={20} style={{ flexShrink: 0, marginTop: 2 }} />
      <div style={{ flex: 1 }}>
        <h4 style={{ margin: 0, fontSize: '0.875rem', fontWeight: 600, color: '#f4f4f5' }}>{toast.title}</h4>
        {toast.message && <p style={{ margin: '4px 0 0', fontSize: '0.8rem', color: '#a1a1aa' }}>{toast.message}</p>}
      </div>
      <button onClick={onRemove} style={{
        background: 'none', border: 'none', color: '#71717a', cursor: 'pointer', padding: 0
      }}>
        <X size={16} />
      </button>
    </div>
  );
}

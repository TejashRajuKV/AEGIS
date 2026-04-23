'use client';
import React from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
  leftAddon?: React.ReactNode;
}

export function Input({ label, error, hint, leftAddon, className = '', style, ...rest }: InputProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem', width: '100%' }}>
      {label && (
        <label style={{ fontSize: '0.8rem', fontWeight: 600, color: '#a1a1aa', letterSpacing: '0.03em' }}>
          {label}
        </label>
      )}
      <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
        {leftAddon && (
          <span style={{
            position: 'absolute', left: '0.75rem',
            color: '#71717a', display: 'flex', alignItems: 'center',
          }}>
            {leftAddon}
          </span>
        )}
        <input
          className={`input-field ${className}`}
          style={{
            paddingLeft: leftAddon ? '2.25rem' : undefined,
            borderColor: error ? '#ef4444' : undefined,
            ...style,
          }}
          {...rest}
        />
      </div>
      {error && <span style={{ fontSize: '0.75rem', color: '#f87171' }}>{error}</span>}
      {hint && !error && <span style={{ fontSize: '0.75rem', color: '#71717a' }}>{hint}</span>}
    </div>
  );
}

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  hint?: string;
}

export function Textarea({ label, error, hint, style, ...rest }: TextareaProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem', width: '100%' }}>
      {label && (
        <label style={{ fontSize: '0.8rem', fontWeight: 600, color: '#a1a1aa', letterSpacing: '0.03em' }}>
          {label}
        </label>
      )}
      <textarea
        className="input-field"
        style={{ resize: 'vertical', minHeight: 100, borderColor: error ? '#ef4444' : undefined, ...style }}
        {...rest}
      />
      {error && <span style={{ fontSize: '0.75rem', color: '#f87171' }}>{error}</span>}
      {hint && !error && <span style={{ fontSize: '0.75rem', color: '#71717a' }}>{hint}</span>}
    </div>
  );
}

'use client';
import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
  maxHeight?: number;
}

export function CodeBlock({ code, language = 'python', filename, maxHeight = 480 }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{
      background: '#0d0d12',
      border: '1px solid rgba(255,255,255,0.06)',
      borderRadius: 'var(--radius-md)',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.625rem 1rem',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        background: 'rgba(255,255,255,0.02)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ display: 'flex', gap: 5 }}>
            {['#ef4444', '#f59e0b', '#22c55e'].map(c => (
              <div key={c} style={{ width: 8, height: 8, borderRadius: '50%', background: c, opacity: 0.5 }} />
            ))}
          </div>
          {filename && <span style={{ fontSize: '0.72rem', color: '#52525b', fontFamily: 'var(--font-mono)' }}>{filename}</span>}
          {!filename && <span style={{ fontSize: '0.72rem', color: '#52525b', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{language}</span>}
        </div>
        <button
          onClick={copy}
          style={{
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: copied ? '#4ade80' : '#52525b',
            display: 'flex', alignItems: 'center', gap: 4,
            fontSize: '0.7rem', transition: 'color 0.2s',
          }}
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>

      {/* Code */}
      <pre style={{
        padding: '1.25rem 1.5rem',
        margin: 0,
        fontFamily: 'var(--font-mono)',
        fontSize: '0.8rem',
        lineHeight: 1.7,
        color: '#e4e4e7',
        overflowY: 'auto',
        overflowX: 'auto',
        maxHeight,
        whiteSpace: 'pre',
      }}>
        <code>{code}</code>
      </pre>
    </div>
  );
}

'use client';
import React from 'react';

interface Column<T> { key: keyof T | string; header: string; render?: (row: T) => React.ReactNode; width?: string; }

interface TableProps<T extends Record<string, unknown>> {
  columns: Column<T>[];
  data: T[];
  emptyMessage?: string;
  stickyHeader?: boolean;
  maxHeight?: number;
}

export function Table<T extends Record<string, unknown>>({
  columns, data, emptyMessage = 'No data', stickyHeader, maxHeight,
}: TableProps<T>) {
  return (
    <div style={{
      borderRadius: 'var(--radius-md)',
      border: '1px solid rgba(255,255,255,0.07)',
      overflow: 'hidden',
      maxHeight: maxHeight ? `${maxHeight}px` : undefined,
      overflowY: maxHeight ? 'auto' : undefined,
    }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-sans)', fontSize: '0.825rem' }}>
        <thead style={{ position: stickyHeader ? 'sticky' : undefined, top: 0, zIndex: 1 }}>
          <tr style={{ background: 'rgba(255,255,255,0.04)' }}>
            {columns.map(col => (
              <th key={String(col.key)} style={{
                padding: '0.75rem 1rem',
                textAlign: 'left',
                fontSize: '0.7rem',
                fontWeight: 700,
                color: '#71717a',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                borderBottom: '1px solid rgba(255,255,255,0.07)',
                width: col.width,
                whiteSpace: 'nowrap',
              }}>
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td colSpan={columns.length} style={{ padding: '2.5rem', textAlign: 'center', color: '#52525b', fontSize: '0.875rem' }}>
                {emptyMessage}
              </td>
            </tr>
          ) : data.map((row, ri) => (
            <tr key={ri} style={{
              borderBottom: '1px solid rgba(255,255,255,0.04)',
              transition: 'background 0.15s',
            }}
            onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.02)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            >
              {columns.map(col => (
                <td key={String(col.key)} style={{ padding: '0.75rem 1rem', color: '#d4d4d8', verticalAlign: 'middle' }}>
                  {col.render ? col.render(row) : String(row[col.key as keyof T] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

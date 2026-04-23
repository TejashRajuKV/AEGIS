'use client';
import React from 'react';

interface SkeletonProps { width?: string | number; height?: string | number; style?: React.CSSProperties; }
export function Skeleton({ width = '100%', height = 16, style }: SkeletonProps) {
  return <div className="skeleton" style={{ width, height, ...style }} />;
}

interface SkeletonCardProps { lines?: number; }
export function SkeletonCard({ lines = 3 }: SkeletonCardProps) {
  return (
    <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.875rem' }}>
      <Skeleton height={14} width="60%" />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} height={11} width={`${90 - i * 10}%`} />
      ))}
    </div>
  );
}

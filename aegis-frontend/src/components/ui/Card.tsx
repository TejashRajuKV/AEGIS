'use client';
import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  onClick?: () => void;
  hoverable?: boolean;
  padding?: string;
}

export function Card({ children, className = '', style = {}, onClick, hoverable, padding = '1.5rem' }: CardProps) {
  const [hover, setHover] = React.useState(false);
  return (
    <div
      className={`glass-panel ${className}`}
      onClick={onClick}
      onMouseEnter={() => hoverable && setHover(true)}
      onMouseLeave={() => hoverable && setHover(false)}
      style={{
        padding,
        cursor: onClick || hoverable ? 'pointer' : undefined,
        transform: hover ? 'translateY(-2px)' : undefined,
        boxShadow: hover ? '0 12px 40px rgba(0,0,0,0.5)' : undefined,
        transition: 'transform 0.2s ease, box-shadow 0.2s ease',
        ...style,
      }}
    >
      {children}
    </div>
  );
}

interface CardHeaderProps { children: React.ReactNode; style?: React.CSSProperties; }
export function CardHeader({ children, style }: CardHeaderProps) {
  return (
    <div style={{ marginBottom: '1.25rem', paddingBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.05)', ...style }}>
      {children}
    </div>
  );
}

interface CardTitleProps { children: React.ReactNode; size?: 'sm' | 'md' | 'lg'; }
export function CardTitle({ children, size = 'md' }: CardTitleProps) {
  const sizes = { sm: '0.9rem', md: '1.1rem', lg: '1.35rem' };
  return (
    <h3 style={{ fontSize: sizes[size], fontWeight: 600, color: '#f4f4f5', fontFamily: 'var(--font-sans)' }}>
      {children}
    </h3>
  );
}

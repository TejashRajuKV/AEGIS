'use client';
import React from 'react';

type Variant = 'primary' | 'outline' | 'ghost' | 'danger';
type Size = 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  children: React.ReactNode;
}

const variantStyles: Record<Variant, string> = {
  primary: 'btn-primary',
  outline: 'btn-outline',
  ghost:   'btn-ghost',
  danger:  '',
};

const sizeMap: Record<Size, React.CSSProperties> = {
  sm: { padding: '0.45rem 0.875rem', fontSize: '0.78rem' },
  md: { padding: '0.625rem 1.375rem', fontSize: '0.875rem' },
  lg: { padding: '0.875rem 1.875rem', fontSize: '1rem' },
};

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  leftIcon,
  rightIcon,
  children,
  style,
  disabled,
  ...rest
}: ButtonProps) {
  const dangerStyle: React.CSSProperties = variant === 'danger'
    ? { background: 'rgba(239,68,68,0.15)', color: '#f87171', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 'var(--radius-md)', fontFamily: 'var(--font-sans)', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: 6, transition: 'all 0.2s', cursor: 'pointer' }
    : {};

  return (
    <button
      className={variant !== 'danger' ? variantStyles[variant] : ''}
      disabled={disabled || loading}
      style={{
        ...sizeMap[size],
        opacity: disabled || loading ? 0.5 : 1,
        pointerEvents: disabled || loading ? 'none' : undefined,
        ...dangerStyle,
        ...style,
      }}
      {...rest}
    >
      {loading && <span className="spinner" style={{ width: 13, height: 13 }} />}
      {!loading && leftIcon}
      {children}
      {!loading && rightIcon}
    </button>
  );
}

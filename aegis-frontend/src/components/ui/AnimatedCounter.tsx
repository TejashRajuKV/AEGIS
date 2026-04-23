'use client';
import { useEffect, useState, useRef } from 'react';

interface AnimatedCounterProps {
  value: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
}

export function AnimatedCounter({ value, duration = 1.5, decimals = 0, prefix = '', suffix = '' }: AnimatedCounterProps) {
  const [current, setCurrent] = useState(0);
  const previousValue = useRef(0);

  useEffect(() => {
    let startTimestamp: number | null = null;
    const startValue = previousValue.current;
    
    // Only animate if there's a difference
    if (startValue === value) {
      setCurrent(value);
      return;
    }

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / (duration * 1000), 1);
      
      // easeOutExpo
      const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      
      setCurrent(startValue + (value - startValue) * easeProgress);

      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        setCurrent(value);
        previousValue.current = value;
      }
    };

    window.requestAnimationFrame(step);
  }, [value, duration]);

  return (
    <span>
      {prefix}
      {current.toFixed(decimals)}
      {suffix}
    </span>
  );
}

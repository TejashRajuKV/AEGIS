'use client';

import { useEffect, useRef } from 'react';

/* ─── Custom magnetic cursor ─── */
export function CustomCursor() {
  const dotRef  = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let mx = 0, my = 0, rx = 0, ry = 0, raf = 0;

    const move = (e: MouseEvent) => { mx = e.clientX; my = e.clientY; };

    const loop = () => {
      rx += (mx - rx) * 0.12;
      ry += (my - ry) * 0.12;
      if (dotRef.current)  { dotRef.current.style.transform  = `translate(${mx}px,${my}px) translate(-50%,-50%)`; }
      if (ringRef.current) { ringRef.current.style.transform = `translate(${rx}px,${ry}px) translate(-50%,-50%)`; }
      raf = requestAnimationFrame(loop);
    };

    const over = (e: MouseEvent) => {
      const t = e.target as HTMLElement;
      const hover = t.closest('a,button,[data-cursor]');
      if (ringRef.current) {
        ringRef.current.style.width  = hover ? '52px' : '34px';
        ringRef.current.style.height = hover ? '52px' : '34px';
        ringRef.current.style.background = hover ? 'rgba(91,169,154,0.12)' : 'transparent';
      }
    };

    window.addEventListener('mousemove', move);
    window.addEventListener('mouseover', over);
    raf = requestAnimationFrame(loop);
    return () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseover', over);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <>
      {/* Inner dot */}
      <div ref={dotRef} style={{ position:'fixed', top:0, left:0, width:7, height:7, borderRadius:'50%', background:'var(--teal)', pointerEvents:'none', zIndex:9999, mixBlendMode:'multiply', transition:'background 0.2s' }} />
      {/* Outer ring */}
      <div ref={ringRef} style={{ position:'fixed', top:0, left:0, width:34, height:34, borderRadius:'50%', border:'1.5px solid var(--teal)', pointerEvents:'none', zIndex:9998, opacity:0.7, transition:'width 0.3s cubic-bezier(.34,1.56,.64,1), height 0.3s cubic-bezier(.34,1.56,.64,1), background 0.3s' }} />
    </>
  );
}

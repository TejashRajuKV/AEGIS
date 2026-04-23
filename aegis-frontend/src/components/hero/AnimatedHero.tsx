'use client';

import { useEffect, useRef } from 'react';

/* ────────────────────────────────────────────
   AnimatedHero — full-screen SVG that draws
   itself on load. Cyberpunk / Neural network
   landscape — nodes, edges, data streams,
   floating "AEGIS orb".
────────────────────────────────────────────── */

export function AnimatedHero() {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const paths = svgRef.current.querySelectorAll('.draw-path');
    paths.forEach((path, i) => {
      const el = path as SVGGeometryElement;
      try {
        const len = el.getTotalLength ? el.getTotalLength() : 800;
        (el as any).style.strokeDasharray = len;
        (el as any).style.strokeDashoffset = len;
        (el as any).style.animation = `none`;
        (el as any).style.transition = 'none';
        requestAnimationFrame(() => {
          (el as any).style.transition = `stroke-dashoffset ${1.2 + i * 0.08}s cubic-bezier(0.4,0,0.2,1) ${0.3 + i * 0.05}s`;
          (el as any).style.strokeDashoffset = '0';
        });
      } catch (_) {}
    });
  }, []);

  return (
    <div style={{ position:'absolute', inset:0, overflow:'hidden', zIndex:0 }}>
      <svg
        ref={svgRef}
        viewBox="0 0 1440 820"
        preserveAspectRatio="xMidYMid slice"
        style={{ width:'100%', height:'100%' }}
        aria-hidden
      >
        {/* ── Background gradient sky ── */}
        <defs>
          <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#050505" />
            <stop offset="60%" stopColor="#0a0f16" />
            <stop offset="100%" stopColor="rgba(0,243,255,0.1)" />
          </linearGradient>
          <linearGradient id="horizonGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(0,243,255,0.2)" />
            <stop offset="100%" stopColor="rgba(255,0,60,0.05)" />
          </linearGradient>
          <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#00f3ff" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#00f3ff" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="nodeGlowMagenta" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#ff003c" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#ff003c" stopOpacity="0" />
          </radialGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="5" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <rect width="1440" height="820" fill="url(#skyGrad)" />

        {/* ── Horizon band ── */}
        <rect y="540" width="1440" height="280" fill="url(#horizonGrad)" opacity="0.6"/>

        {/* ── Ground plane — circuit board style ── */}
        {/* Main grid lines — horizontal */}
        {[560,600,640,680,720,760,800].map((y,i) => (
          <line key={`hg${i}`} className="draw-path" x1={-50} y1={y + i*8} x2={1490} y2={y + i*5}
            stroke="rgba(0,243,255,0.25)" strokeWidth="1" fill="none" />
        ))}
        {/* Main grid lines — converging perspective */}
        {[0,80,160,240,320,400,480,560,640,720,800,880,960,1040,1120,1200,1280,1360,1440].map((x,i) => (
          <line key={`vg${i}`} className="draw-path" x1={x} y1={820} x2={720} y2={540}
            stroke="rgba(0,243,255,0.15)" strokeWidth="1" fill="none" />
        ))}

        {/* ── Main circuit trace — left backbone ── */}
        <path className="draw-path" d="M 0 700 L 200 700 L 200 640 L 380 640 L 380 590 L 520 590 L 520 555 L 650 555 L 650 540 L 720 540"
          stroke="#00f3ff" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round" filter="url(#glow)"/>

        {/* ── Main circuit trace — right backbone ── */}
        <path className="draw-path" d="M 1440 700 L 1240 700 L 1240 640 L 1060 640 L 1060 590 L 920 590 L 920 555 L 790 555 L 790 540 L 720 540"
          stroke="#00f3ff" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round" filter="url(#glow)"/>

        {/* ── Branch traces — left ── */}
        <path className="draw-path" d="M 200 700 L 200 760 L 60 760" stroke="#00f3ff" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.8"/>
        <path className="draw-path" d="M 380 640 L 280 640 L 280 710 L 140 710" stroke="#00f3ff" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.7"/>
        <path className="draw-path" d="M 520 590 L 420 590 L 420 670 L 320 670" stroke="#ff003c" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.8" filter="url(#glow)"/>
        <path className="draw-path" d="M 520 590 L 520 680 L 450 680 L 450 740" stroke="#ff003c" strokeWidth="1.2" fill="none" strokeLinecap="round" opacity="0.6"/>
        <path className="draw-path" d="M 650 555 L 580 555 L 580 650 L 480 650" stroke="#00f3ff" strokeWidth="1.2" fill="none" strokeLinecap="round" opacity="0.6"/>

        {/* ── Branch traces — right ── */}
        <path className="draw-path" d="M 1240 700 L 1240 760 L 1380 760" stroke="#00f3ff" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.8"/>
        <path className="draw-path" d="M 1060 640 L 1160 640 L 1160 710 L 1300 710" stroke="#00f3ff" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.7"/>
        <path className="draw-path" d="M 920 590 L 1020 590 L 1020 670 L 1120 670" stroke="#ff003c" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.8" filter="url(#glow)"/>
        <path className="draw-path" d="M 920 590 L 920 680 L 990 680 L 990 740" stroke="#ff003c" strokeWidth="1.2" fill="none" strokeLinecap="round" opacity="0.6"/>
        <path className="draw-path" d="M 790 555 L 860 555 L 860 650 L 960 650" stroke="#00f3ff" strokeWidth="1.2" fill="none" strokeLinecap="round" opacity="0.6"/>

        {/* ── Rising structures (data pillars) — left ── */}
        <path className="draw-path" d="M 80 820 L 80 650 L 60 630 L 80 610 L 100 630 L 80 650" stroke="rgba(0,243,255,0.5)" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" filter="url(#glow)"/>
        <path className="draw-path" d="M 80 630 L 80 550 L 55 520 L 80 490 L 105 520 L 80 550" stroke="rgba(0,243,255,0.4)" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
        <path className="draw-path" d="M 80 500 L 80 430 L 60 400 L 80 370 L 100 400 L 80 430" stroke="rgba(0,243,255,0.3)" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
        
        <path className="draw-path" d="M 180 820 L 180 680 L 160 660 L 180 640 L 200 660 L 180 680" stroke="rgba(255,0,60,0.5)" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" filter="url(#glow)"/>
        <path className="draw-path" d="M 180 645 L 180 560 L 158 535 L 180 510 L 202 535 L 180 560" stroke="rgba(255,0,60,0.4)" strokeWidth="1.6" fill="none" strokeLinecap="round" strokeLinejoin="round"/>

        <path className="draw-path" d="M 320 820 L 320 700 L 296 676 L 320 652 L 344 676 L 320 700" stroke="rgba(0,243,255,0.5)" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
        <path className="draw-path" d="M 320 655 L 320 570 L 300 548 L 320 526 L 340 548 L 320 570" stroke="rgba(0,243,255,0.4)" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>

        {/* ── Rising structures — right ── */}
        <path className="draw-path" d="M 1360 820 L 1360 650 L 1380 630 L 1360 610 L 1340 630 L 1360 650" stroke="rgba(0,243,255,0.5)" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" filter="url(#glow)"/>
        <path className="draw-path" d="M 1360 630 L 1360 550 L 1385 520 L 1360 490 L 1335 520 L 1360 550" stroke="rgba(0,243,255,0.4)" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
        
        <path className="draw-path" d="M 1260 820 L 1260 680 L 1280 660 L 1260 640 L 1240 660 L 1260 680" stroke="rgba(255,0,60,0.5)" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" filter="url(#glow)"/>
        <path className="draw-path" d="M 1260 645 L 1260 560 L 1282 535 L 1260 510 L 1238 535 L 1260 560" stroke="rgba(255,0,60,0.4)" strokeWidth="1.6" fill="none" strokeLinecap="round" strokeLinejoin="round"/>

        <path className="draw-path" d="M 1120 820 L 1120 700 L 1144 676 L 1120 652 L 1096 676 L 1120 700" stroke="rgba(0,243,255,0.5)" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
        <path className="draw-path" d="M 1120 655 L 1120 570 L 1140 548 L 1120 526 L 1100 548 L 1120 570" stroke="rgba(0,243,255,0.4)" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>

        {/* ── Foreground circuit nodes (on the ground plane) ── */}
        {[
          [200,700],[380,640],[520,590],[650,555],[790,555],[920,590],[1060,640],[1240,700],
          [280,710],[420,590],[580,555],[860,650],[1020,670],[1160,640],
        ].map(([cx,cy],i) => {
          const isMagenta = i === 2 || i === 5 || i === 9 || i === 12;
          const color = isMagenta ? '#ff003c' : '#00f3ff';
          const glow = isMagenta ? 'url(#nodeGlowMagenta)' : 'url(#nodeGlow)';
          return (
            <g key={`fn${i}`}>
              <circle cx={cx} cy={cy} r="16" fill={glow} opacity="0.9"/>
              <circle cx={cx} cy={cy} r="6" fill="#050505" stroke={color} strokeWidth="2"/>
              <circle cx={cx} cy={cy} r="3" fill={color} filter="url(#glow)"/>
            </g>
          )
        })}

        {/* ── Central vanishing point — AEGIS hub ── */}
        <circle cx="720" cy="540" r="50" fill="url(#nodeGlow)" opacity="0.8"/>
        <circle cx="720" cy="540" r="32" fill="#0a0f16" stroke="#00f3ff" strokeWidth="2" filter="url(#glow)"/>
        <circle cx="720" cy="540" r="42" fill="none" stroke="#00f3ff" strokeWidth="1.5" strokeDasharray="4 6" opacity="0.7" style={{animation:'orbSpin 8s linear infinite',transformOrigin:'720px 540px'}}/>
        <circle cx="720" cy="540" r="16" fill="#00f3ff" filter="url(#glow)" opacity="0.9"/>
        <text x="720" y="544" textAnchor="middle" fontSize="10" fill="#050505" fontFamily="var(--font-orbitron)" fontWeight="700" letterSpacing="1">AEGIS</text>

        {/* ── Floating AEGIS orb — like the balloon ── */}
        <g style={{animation:'floatBalloon 6s ease-in-out infinite',transformOrigin:'720px 280px'}}>
          {/* Orb */}
          <circle cx="720" cy="280" r="60" fill="url(#nodeGlow)" opacity="0.9" filter="url(#glow)"/>
          <circle cx="720" cy="280" r="45" fill="rgba(0,243,255,0.1)" stroke="#00f3ff" strokeWidth="2"/>
          <circle cx="720" cy="280" r="55" fill="none" stroke="#00f3ff" strokeWidth="1.5" strokeDasharray="5 10" opacity="0.8" style={{animation:'orbSpin 10s linear infinite',transformOrigin:'720px 280px'}}/>
          {/* Inner rings */}
          <circle cx="720" cy="280" r="35" fill="none" stroke="#00f3ff" strokeWidth="1" opacity="0.5"/>
          <circle cx="720" cy="280" r="25" fill="none" stroke="#00f3ff" strokeWidth="1" opacity="0.3" strokeDasharray="2 4" style={{animation:'orbSpin 5s linear infinite reverse',transformOrigin:'720px 280px'}}/>
          {/* Label */}
          <text x="720" y="275" textAnchor="middle" fontSize="14" fill="#fff" fontFamily="var(--font-orbitron)" fontWeight="700" letterSpacing="2" filter="url(#glow)">AEGIS</text>
          <text x="720" y="292" textAnchor="middle" fontSize="8" fill="#00f3ff" fontFamily="var(--font-space-grotesk)" letterSpacing="1.5">AI FAIRNESS</text>
          {/* Tether line */}
          <line x1="720" y1="335" x2="720" y2="508" stroke="#00f3ff" strokeWidth="1.5" strokeDasharray="6 4" opacity="0.6" filter="url(#glow)"/>
        </g>

        {/* ── Ambient data particles — small dots on circuit lines ── */}
        {[
          {x:400,y:640,delay:0, col:'#ff003c'}, {x:560,y:590,delay:0.5, col:'#ff003c'}, {x:680,y:555,delay:1.0, col:'#00f3ff'},
          {x:760,y:555,delay:0.3, col:'#00f3ff'}, {x:880,y:590,delay:0.8, col:'#00f3ff'}, {x:1000,y:640,delay:1.3, col:'#ff003c'},
          {x:300,y:700,delay:0.2, col:'#00f3ff'}, {x:450,y:670,delay:0.7, col:'#ff003c'}, {x:1140,y:670,delay:0.4, col:'#00f3ff'},
        ].map((p,i) => (
          <circle key={`p${i}`} cx={p.x} cy={p.y} r="4" fill={p.col} opacity="0" filter="url(#glow)"
            style={{animation:`fadeIn 0.5s ${p.delay + 2.5}s ease forwards`,animationFillMode:'both'}} />
        ))}

        {/* ── Corner labels — module hints ── */}
        {[
          {x:80, y:360, label:'CAUSAL\nDISCOVERY', anchor:'middle'},
          {x:180, y:500, label:'DRIFT\nMONITOR', anchor:'middle'},
          {x:1360,y:360, label:'RL\nAUTOPILOT', anchor:'middle'},
          {x:1260,y:500, label:'TEXT\nBIAS', anchor:'middle'},
        ].map((l,i) => (
          <text key={`lbl${i}`} x={l.x} y={l.y} textAnchor={l.anchor as any} fontSize="10"
            fill="#00f3ff" fontFamily="var(--font-orbitron)" fontWeight="600" letterSpacing="1"
            style={{animation:`fadeIn 0.6s ${2.8 + i*0.2}s ease both`,animationFillMode:'both', textShadow:'0 0 5px rgba(0,243,255,0.5)'}}>
            {l.label.split('\n').map((t,j) => <tspan key={j} x={l.x} dy={j===0?0:14}>{t}</tspan>)}
          </text>
        ))}
      </svg>

      {/* Scan line effect */}
      <div className="scan-line" style={{opacity:0.8}}/>
    </div>
  );
}

/* Compass that spins — futuristic version */
export function CompassOrnament() {
  return (
    <svg viewBox="0 0 80 80" style={{width:80,height:80}}>
      <circle cx="40" cy="40" r="36" fill="none" stroke="rgba(0,243,255,0.3)" strokeWidth="2"/>
      <circle cx="40" cy="40" r="30" fill="rgba(5,5,5,0.8)" stroke="rgba(0,243,255,0.5)" strokeWidth="1"/>
      {/* Compass rose */}
      <g style={{animation:'spinCompass 6s ease-in-out infinite',transformOrigin:'40px 40px'}}>
        <polygon points="40,14 43,40 40,38 37,40" fill="#00f3ff"/>
        <polygon points="40,66 37,40 40,42 43,40" fill="rgba(0,243,255,0.3)"/>
        <polygon points="14,40 40,37 38,40 40,43" fill="rgba(0,243,255,0.3)"/>
        <polygon points="66,40 40,43 42,40 40,37" fill="rgba(0,243,255,0.3)"/>
      </g>
      <circle cx="40" cy="40" r="4" fill="#00f3ff" filter="drop-shadow(0 0 3px #00f3ff)"/>
      <text x="40" y="10" textAnchor="middle" fontSize="8" fill="#00f3ff" fontFamily="var(--font-orbitron)" fontWeight="bold">N</text>
    </svg>
  );
}

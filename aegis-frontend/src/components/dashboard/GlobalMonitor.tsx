'use client';
import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Sphere, Stars, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

function Globe() {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (meshRef.current) {
      meshRef.current.rotation.y = t * 0.1;
    }
    if (glowRef.current) {
      glowRef.current.scale.setScalar(1.1 + Math.sin(t * 2) * 0.02);
    }
  });

  return (
    <group>
      <Sphere ref={meshRef} args={[1, 64, 64]}>
        <meshStandardMaterial 
          color="#6366f1" 
          wireframe 
          transparent 
          opacity={0.3} 
          blending={THREE.AdditiveBlending} 
        />
      </Sphere>
      <Sphere ref={glowRef} args={[1.05, 32, 32]}>
        <meshBasicMaterial 
          color="#6366f1" 
          transparent 
          opacity={0.05} 
          side={THREE.BackSide} 
        />
      </Sphere>
      
      {/* Random "Audit" Pings */}
      {[...Array(6)].map((_, i) => (
        <Ping key={i} delay={i * 2} />
      ))}
    </group>
  );
}

function Ping({ delay }: { delay: number }) {
  const ref = useRef<THREE.Mesh>(null);
  const pos = useMemo(() => {
    const r = 1;
    const theta = Math.random() * 2 * Math.PI;
    const phi = Math.acos(2 * Math.random() - 1);
    return [
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi)
    ];
  }, []);

  useFrame((state) => {
    const t = (state.clock.getElapsedTime() + delay) % 4;
    if (ref.current) {
      ref.current.scale.setScalar(t * 0.5);
      if (ref.current.material instanceof THREE.Material) {
        ref.current.material.opacity = Math.max(0, 1 - t / 4) * 0.5;
      }
    }
  });

  return (
    <mesh position={pos as any} ref={ref}>
      <sphereGeometry args={[0.1, 16, 16]} />
      <meshBasicMaterial color="#818cf8" transparent opacity={0.5} />
    </mesh>
  );
}

export function GlobalMonitor() {
  return (
    <div className="w-full h-full min-h-[300px] relative">
      <Canvas dpr={[1, 2]}>
        <PerspectiveCamera makeDefault position={[0, 0, 2.5]} fov={45} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#818cf8" />
        <Globe />
      </Canvas>
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-[#050508] via-transparent to-transparent opacity-60" />
    </div>
  );
}

'use client';

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/* ──────────────────────────────────────────
   Custom shader sphere — looks like a
   glowing plasma / holographic orb
────────────────────────────────────────── */
const vertexShader = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform float uTime;

  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;

    // subtle vertex displacement
    vec3 pos = position;
    float displacement = sin(pos.x * 4.0 + uTime * 1.5) *
                         sin(pos.y * 4.0 + uTime * 1.2) *
                         sin(pos.z * 4.0 + uTime * 0.8) * 0.04;
    pos += normal * displacement;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const fragmentShader = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform float uTime;

  void main() {
    vec3 col1 = vec3(0.9, 0.9, 0.95);     // light grey/white
    vec3 col2 = vec3(0.5, 0.5, 0.55);     // mid grey
    vec3 col3 = vec3(0.2, 0.2, 0.25);     // dark grey

    float fresnel = pow(1.0 - abs(dot(vNormal, vec3(0.0,0.0,1.0))), 2.5);
    float pulse = sin(uTime * 0.8) * 0.5 + 0.5;
    float stripe = sin(vPosition.y * 12.0 + uTime * 2.0) * 0.5 + 0.5;

    vec3 color = mix(col1, col2, fresnel);
    color = mix(color, col3, stripe * 0.15);
    color += fresnel * 0.5 * mix(col2, col3, pulse);

    float alpha = 0.85 + fresnel * 0.15;
    gl_FragColor = vec4(color, alpha);
  }
`;

export function ShaderOrb() {
  const matRef = useRef<THREE.ShaderMaterial>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const uniforms = useMemo(() => ({ uTime: { value: 0 } }), []);

  useFrame(({ clock }) => {
    uniforms.uTime.value = clock.getElapsedTime();
    if (meshRef.current) {
      meshRef.current.rotation.y = clock.getElapsedTime() * 0.12;
      meshRef.current.rotation.x = Math.sin(clock.getElapsedTime() * 0.07) * 0.15;
    }
  });

  return (
    <group>
      {/* Outer glow sphere */}
      <mesh scale={1.35}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.02} />
      </mesh>
      <mesh scale={1.15}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshBasicMaterial color="#a1a1aa" transparent opacity={0.03} />
      </mesh>
      {/* Main shader sphere */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[1, 128, 128]} />
        <shaderMaterial
          ref={matRef}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          uniforms={uniforms}
          transparent
        />
      </mesh>
      {/* Wireframe overlay */}
      <mesh>
        <sphereGeometry args={[1.01, 24, 24]} />
        <meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.04} />
      </mesh>
    </group>
  );
}

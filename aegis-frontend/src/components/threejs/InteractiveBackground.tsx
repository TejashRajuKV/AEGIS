'use client';

import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { ParticleGalaxy, NeuralWeb, OrbitalSystem } from './ParticleGalaxy';
import { ShaderOrb } from './ShaderOrb';
import { SceneLighting } from './SceneLighting';

function CameraRig() {
  const { camera, pointer } = useThree();
  const vec = new THREE.Vector3();

  useFrame(() => {
    // Reduced sensitivity (0.8 vs old 2.0) + slower lerp (0.03 vs 0.05) = silky parallax
    camera.position.lerp(vec.set(pointer.x * 0.8, pointer.y * 0.8, 5), 0.03);
    camera.lookAt(0, 0, 0);
  });
  return null;
}

export function InteractiveBackground() {
  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none', background: 'var(--background)' }}>
      <Canvas camera={{ position: [0, 0, 5], fov: 60 }} dpr={[1, 1.5]}>
        <SceneLighting />

        {/* Orb: shifted RIGHT to align with hero right column */}
        <group position={[2.2, -0.3, -1.5]}>
          <ShaderOrb />
        </group>

        {/* Particle galaxy: wide spread across full canvas, pushed back */}
        <group position={[0, 0, -5]} scale={1.6}>
          <ParticleGalaxy />
        </group>

        {/* Neural web: upper-left, provides depth contrast to the orb */}
        <group position={[-2.5, 1.2, -2]} scale={1.2}>
          <NeuralWeb />
        </group>

        {/* Subtle orbital rings around the orb */}
        <group position={[2.2, -0.3, -1.5]}>
          <OrbitalSystem
            radius={1.8} count={4} tilt={[0.4, 0, 0.2]}
            speed={0.18} color="#6366f1" nodeSize={0.04}
          />
          <OrbitalSystem
            radius={2.4} count={6} tilt={[-0.2, 0.3, 0]}
            speed={-0.12} color="#818cf8" nodeSize={0.03}
          />
        </group>

        <CameraRig />
      </Canvas>
    </div>
  );
}

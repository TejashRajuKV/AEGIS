'use client';
import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  Float, 
  MeshTransmissionMaterial, 
  Sphere, 
  Stars, 
  Html,
  PerspectiveCamera,
  Torus,
  Points,
  PointMaterial,
} from '@react-three/drei';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';

/* ── Floating Neural Particles ─────────────────────────── */
function NeuralParticles({ count = 2000 }) {
  const points = useMemo(() => {
    const p = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 2.5 + Math.random() * 2;
      const theta = Math.random() * 2 * Math.PI;
      const phi = Math.acos(2 * Math.random() - 1);
      p[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      p[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      p[i * 3 + 2] = r * Math.cos(phi);
    }
    return p;
  }, [count]);

  const ref = useRef<THREE.Points>(null);
  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = state.clock.getElapsedTime() * 0.05;
      ref.current.rotation.z = state.clock.getElapsedTime() * 0.03;
    }
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={points.length / 3}
          array={points}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial 
        size={0.015} 
        color="#818cf8" 
        transparent 
        opacity={0.4} 
        blending={THREE.AdditiveBlending} 
        sizeAttenuation 
      />
    </points>
  );
}

/* ── Orbiting Data Node ────────────────────────────────── */
function OrbitingNode({ radius, speed, label, value, color, delay }: { radius: number; speed: number; label: string; value: string; color: string; delay: number }) {
  const ref = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (ref.current) {
      const t = state.clock.getElapsedTime() * speed + delay;
      ref.current.position.x = Math.cos(t) * radius;
      ref.current.position.z = Math.sin(t) * radius;
      ref.current.position.y = Math.sin(t * 0.5) * (radius * 0.2);
    }
  });

  return (
    <group ref={ref}>
      <mesh onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color} 
          emissiveIntensity={hovered ? 8 : 3} 
          toneMapped={false}
        />
      </mesh>
      <Html distanceFactor={10} position={[0, 0.25, 0]}>
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: hovered ? 1 : 0.6, scale: hovered ? 1.1 : 0.9 }}
          className="whitespace-nowrap flex flex-col gap-0.5 px-3 py-2 bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl text-center pointer-events-none select-none shadow-2xl"
        >
          <span className="text-[8px] font-mono font-bold tracking-[0.2em] text-indigo-300 uppercase opacity-70">{label}</span>
          <span className="text-sm font-serif font-bold text-white">{value}</span>
        </motion.div>
      </Html>
    </group>
  );
}

/* ── Volumetric Energy Rays ────────────────────────────── */
function EnergyRays() {
  const rays = useMemo(() => {
    return [...Array(12)].map((_, i) => ({
      rotation: [Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI],
      length: 4 + Math.random() * 4,
      speed: 0.1 + Math.random() * 0.1
    }));
  }, []);

  return (
    <group>
      {rays.map((ray, i) => (
        <EnergyRay key={i} {...ray} />
      ))}
    </group>
  );
}

function EnergyRay({ rotation, length, speed }: any) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y += speed * 0.05;
      const s = 1 + Math.sin(state.clock.getElapsedTime() * speed * 2) * 0.3;
      ref.current.scale.set(s, 1, s);
    }
  });

  return (
    <group rotation={rotation as any}>
      <mesh ref={ref}>
        <cylinderGeometry args={[0, 0.15, length, 16, 1, true]} />
        <meshBasicMaterial 
          color="#6366f1" 
          transparent 
          opacity={0.03} 
          blending={THREE.AdditiveBlending} 
          side={THREE.DoubleSide} 
        />
      </mesh>
    </group>
  );
}

/* ── Internal Neural Core ──────────────────────────────── */
function InternalBrain() {
  const count = 150;
  const positions = useMemo(() => {
    const p = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 0.85 * Math.pow(Math.random(), 0.5);
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      p[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      p[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      p[i * 3 + 2] = r * Math.cos(phi);
    }
    return p;
  }, []);

  const ref = useRef<THREE.Points>(null);
  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = state.clock.getElapsedTime() * 0.15;
    }
  });

  return (
    <group>
      <Points ref={ref} positions={positions} stride={3}>
        <PointMaterial 
          transparent 
          color="#ffffff" 
          size={0.015} 
          sizeAttenuation={true} 
          depthWrite={false} 
          blending={THREE.AdditiveBlending} 
        />
      </Points>
    </group>
  );
}

/* ── Master AI Core Content ─────────────────────────────── */
export function NeuralCoreContent() {
  const coreRef = useRef<THREE.Group>(null);
  const shellRef = useRef<THREE.Mesh>(null);
  const { mouse, viewport } = useThree();

  useFrame((state) => {
    if (coreRef.current) {
      const targetX = (-mouse.y * viewport.height) / 20;
      const targetY = (mouse.x * viewport.width) / 20;
      coreRef.current.rotation.x = THREE.MathUtils.lerp(coreRef.current.rotation.x, targetX, 0.05);
      coreRef.current.rotation.y = THREE.MathUtils.lerp(coreRef.current.rotation.y, targetY, 0.05);
    }
    if (shellRef.current) {
      shellRef.current.rotation.y -= 0.005;
      shellRef.current.rotation.z += 0.002;
    }
  });

  return (
    <group ref={coreRef}>
      <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
        {/* Main Crystal Sphere */}
        <Sphere args={[1.2, 64, 64]}>
          <MeshTransmissionMaterial
            backside
            samples={16}
            thickness={1}
            roughness={0.05}
            transmission={1}
            ior={1.4}
            chromaticAberration={0.06}
            anisotropy={0.1}
            distortion={0.2}
            distortionScale={0.3}
            temporalDistortion={0.1}
            color="#ffffff"
            attenuationDistance={0.5}
            attenuationColor="#6366f1"
          />
        </Sphere>
        
        {/* Internal Systems */}
        <InternalBrain />
        <EnergyRays />

        {/* Outer Wireframe Shell */}
        <mesh ref={shellRef}>
          <icosahedronGeometry args={[1.8, 2]} />
          <meshBasicMaterial color="#4f46e5" wireframe transparent opacity={0.1} />
        </mesh>

        {/* Governance Rings */}
        <Torus args={[2, 0.005, 16, 100]} rotation={[Math.PI / 2, 0, 0]}>
          <meshBasicMaterial color="#6366f1" transparent opacity={0.2} />
        </Torus>
      </Float>

      {/* Orbiting Interaction Nodes */}
      <OrbitingNode radius={2.2} speed={0.4} label="Bias Check" value="Pass" color="#22c55e" delay={0} />
      <OrbitingNode radius={2.8} speed={0.3} label="Fairness" value="98.2" color="#818cf8" delay={2} />
      <OrbitingNode radius={2.5} speed={0.5} label="Latency" value="0.2ms" color="#6366f1" delay={4} />
    </group>
  );
}

/* ── Full NeuralCore Component ──────────────────────────── */
export function NeuralCore() {
  return (
    <div className="w-full h-[700px] relative cursor-grab active:cursor-grabbing">
      <Canvas dpr={[1, 2]}>
        <PerspectiveCamera makeDefault position={[0, 0, 6]} fov={45} />
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={1.5} color="#818cf8" />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#c084fc" />
        
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        <NeuralParticles />
        <NeuralCoreContent />
      </Canvas>

      {/* Floating HUD Elements */}
      <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
        <motion.div 
          animate={{ rotate: 360 }}
          transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
          className="absolute w-[500px] h-[500px] border-[1px] border-indigo-500/10 rounded-full border-dashed opacity-40" 
        />
        <motion.div 
          animate={{ rotate: -360 }}
          transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
          className="absolute w-[600px] h-[600px] border-[1px] border-purple-500/10 rounded-full border-dashed opacity-20" 
        />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] bg-indigo-500/10 blur-[120px] rounded-full" />
      </div>

      {/* Legend / Status Overlay */}
      <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex items-center gap-8 px-8 py-4 bg-white/[0.02] backdrop-blur-3xl border border-white/5 rounded-2xl">
        {[
          { label: 'Neural Uplink', value: 'Active', color: 'bg-emerald-500' },
          { label: 'Integrity', value: 'High', color: 'bg-indigo-500' },
          { label: 'Latency', value: '0.2ms', color: 'bg-indigo-400' },
        ].map((item, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className={`w-1.5 h-1.5 rounded-full ${item.color} shadow-[0_0_8px_currentColor]`} />
            <div className="flex flex-col">
              <span className="text-[9px] font-mono text-zinc-500 uppercase tracking-widest leading-none mb-1">{item.label}</span>
              <span className="text-xs font-bold text-white leading-none">{item.value}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

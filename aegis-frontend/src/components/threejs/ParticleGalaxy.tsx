'use client';

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/* Massive particle galaxy — 3000 particles with velocity fields */
export function ParticleGalaxy() {
  const pts = useRef<THREE.Points>(null);

  const { positions, velocities, colors } = useMemo(() => {
    const count = 3000;
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);

    const teal   = new THREE.Color('#e4e4e7'); // zinc-200
    const amber  = new THREE.Color('#a1a1aa'); // zinc-400
    const sage   = new THREE.Color('#71717a'); // zinc-500
    const cream  = new THREE.Color('#ffffff'); // white
    const palette = [teal, amber, sage, cream];

    for (let i = 0; i < count; i++) {
      // Distribute in a disc shape (galaxy arm)
      const arm = Math.floor(Math.random() * 3);
      const armAngle = (arm / 3) * Math.PI * 2;
      const radius = 1.5 + Math.random() * 5.5;
      const spread = Math.random() * 0.8;
      const angle = armAngle + radius * 0.4 + (Math.random() - 0.5) * spread;

      pos[i*3]   = Math.cos(angle) * radius + (Math.random()-0.5)*0.5;
      pos[i*3+1] = (Math.random()-0.5) * 0.8;
      pos[i*3+2] = Math.sin(angle) * radius + (Math.random()-0.5)*0.5;

      vel[i*3]   = (Math.random()-0.5)*0.0005;
      vel[i*3+1] = (Math.random()-0.5)*0.0002;
      vel[i*3+2] = (Math.random()-0.5)*0.0005;

      const c = palette[Math.floor(Math.random() * palette.length)];
      col[i*3]   = c.r;
      col[i*3+1] = c.g;
      col[i*3+2] = c.b;
    }
    return { positions: pos, velocities: vel, colors: col };
  }, []);

  useFrame(({ clock }) => {
    if (!pts.current) return;
    const t = clock.getElapsedTime();
    const arr = pts.current.geometry.attributes.position.array as Float32Array;
    for (let i = 0; i < arr.length / 3; i++) {
      // Slight circular drift
      const x = arr[i*3], z = arr[i*3+2];
      const r = Math.sqrt(x*x+z*z);
      const speed = 0.00015 / (r * 0.3 + 0.1);
      const cos = Math.cos(speed), sin = Math.sin(speed);
      arr[i*3]   = x*cos - z*sin;
      arr[i*3+2] = x*sin + z*cos;
      arr[i*3+1] += Math.sin(t*0.3 + i*0.01) * 0.0005;
    }
    pts.current.geometry.attributes.position.needsUpdate = true;
    pts.current.rotation.y = t * 0.03;
  });

  return (
    <points ref={pts}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.025} vertexColors transparent opacity={0.5} sizeAttenuation />
    </points>
  );
}

/* Orbiting ring with glowing nodes */
export function OrbitalSystem({ radius, count, tilt, speed, color, nodeSize }:{
  radius:number; count:number; tilt:[number,number,number]; speed:number; color:string; nodeSize:number;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const col = useMemo(() => new THREE.Color(color), [color]);
  const glowCol = useMemo(() => new THREE.Color(color), [color]);

  const nodeAngles = useMemo(
    () => Array.from({length:count}, (_,i) => (i/count)*Math.PI*2),
    [count]
  );

  useFrame(({ clock }) => {
    if (groupRef.current) groupRef.current.rotation.y = clock.getElapsedTime() * speed;
  });

  return (
    <group ref={groupRef} rotation={tilt}>
      {/* Ring */}
      <mesh>
        <torusGeometry args={[radius, 0.006, 8, 128]} />
        <meshBasicMaterial color={col} transparent opacity={0.10} />
      </mesh>
      {/* Nodes */}
      {nodeAngles.map((angle, i) => {
        const x = Math.cos(angle)*radius, z = Math.sin(angle)*radius;
        return (
          <group key={i} position={[x, 0, z]}>
            <mesh>
              <sphereGeometry args={[nodeSize, 12, 12]} />
              <meshStandardMaterial color={col} emissive={glowCol} emissiveIntensity={0.4} metalness={0.5} roughness={0.2} />
            </mesh>
            <mesh scale={3}>
              <sphereGeometry args={[nodeSize, 6, 6]} />
              <meshBasicMaterial color={glowCol} transparent opacity={0.03} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

/* Animated connection lines between floating nodes */
export function NeuralWeb() {
  const nodes: [number,number,number][] = useMemo(() => [
    [2.5, 0.8, 1.0], [-2.2, 1.2, -0.8], [1.8, -1.0, 2.0],
    [-1.5, -0.5, -2.0], [0.5, 2.0, -1.5], [-0.8, -1.8, 1.2],
    [2.8, 0.2, -1.0], [-2.5, 0.5, 0.8],
  ], []);

  const connections = useMemo(() => {
    const pairs = [];
    for (let i=0;i<nodes.length;i++)
      for (let j=i+1;j<nodes.length;j++)
        if (Math.random() > 0.55) pairs.push([i,j]);
    return pairs;
  }, [nodes]);

  const lineGeos = useMemo(() => connections.map(([i,j]) => {
    const geo = new THREE.BufferGeometry();
    const pts = [new THREE.Vector3(...nodes[i]), new THREE.Vector3(...nodes[j])];
    geo.setFromPoints(pts);
    return geo;
  }), [connections, nodes]);

  const groupRef = useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = clock.getElapsedTime() * 0.06;
      groupRef.current.rotation.x = Math.sin(clock.getElapsedTime()*0.04)*0.1;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Connection lines */}
      {lineGeos.map((geo, i) => (
        <primitive key={`l${i}`} object={new THREE.Line(geo,
          new THREE.LineBasicMaterial({ color:'#a1a1aa', transparent:true, opacity:0.1 })
        )} />
      ))}
      {/* Floating nodes */}
      {nodes.map((pos, i) => (
        <group key={`n${i}`} position={pos}>
          <mesh>
            <sphereGeometry args={[0.08+Math.random()*0.06, 16, 16]} />
            <meshStandardMaterial
              color={i%3===0?'#ffffff':i%3===1?'#e4e4e7':'#a1a1aa'}
              emissive={i%3===0?'#ffffff':i%3===1?'#e4e4e7':'#a1a1aa'}
              emissiveIntensity={0.3}
              metalness={0.6}
              roughness={0.2}
            />
          </mesh>
        </group>
      ))}
    </group>
  );
}

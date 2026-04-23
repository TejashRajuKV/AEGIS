'use client';
import { NeuralCore } from '@/components/threejs/NeuralCore';

/** Isolated Three.js canvas — dynamically imported to avoid blocking FCP */
export default function ThreeHero() {
  return (
    <div className="w-full h-full">
      <NeuralCore />
    </div>
  );
}

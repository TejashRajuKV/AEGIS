import type { Metadata } from 'next';
import { HeroSection } from '@/components/sections/HeroSection';
import { ModulesSection } from '@/components/sections/ModulesSection';
import { ContentSections } from '@/components/sections/ContentSections';

export const metadata: Metadata = {
  title: 'AEGIS — Beyond Human Oversight',
  description:
    'The first autonomous governance layer for AI. AEGIS ensures precision, ethics, and stability across every model deployment — live, in real time.',
};

export default function HomePage() {
  return (
    <>
      <HeroSection />
      <ModulesSection />
      <ContentSections />
    </>
  );
}

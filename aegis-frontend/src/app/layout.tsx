import type { Metadata } from 'next';
import './globals.css';
import { Navbar } from '@/components/layout/Navbar';

export const metadata: Metadata = {
  title: {
    default: 'AEGIS — AI Fairness Platform',
    template: '%s | AEGIS',
  },
  description:
    'The autonomous AI fairness governance platform. Detect bias, run causal discovery, deploy PPO autopilot corrections, monitor drift — in real time.',
  keywords: ['AI fairness', 'bias detection', 'causal discovery', 'machine learning', 'AEGIS'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body style={{ background: 'var(--bg)', color: 'var(--text-primary)' }}>
        <Navbar />
        <main style={{ paddingTop: 60, minHeight: '100vh' }}>
          {children}
        </main>
      </body>
    </html>
  );
}

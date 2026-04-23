'use client';
import Link from 'next/link';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Scale, GitBranch, Activity, Bot, Code2, Shuffle, FileText } from 'lucide-react';

const ACTIONS = [
  { href: '/fairness-audit', label: 'Run Audit', icon: Scale, color: '#6366f1' },
  { href: '/causal-graph', label: 'Causal Graph', icon: GitBranch, color: '#f59e0b' },
  { href: '/drift-monitor', label: 'Drift Monitor', icon: Activity, color: '#ef4444' },
  { href: '/autopilot', label: 'Launch Autopilot', icon: Bot, color: '#8b5cf6' },
  { href: '/code-fix', label: 'Code Fix Generator', icon: Code2, color: '#10b981' },
  { href: '/counterfactual', label: 'Counterfactuals', icon: Shuffle, color: '#06b6d4' },
  { href: '/text-bias', label: 'Text Bias Scan', icon: FileText, color: '#ec4899' },
];

export function QuickActionsGrid() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Quick Actions</CardTitle>
      </CardHeader>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '1rem' }}>
        {ACTIONS.map(action => (
          <Link key={action.href} href={action.href} style={{ textDecoration: 'none' }}>
            <div style={{ 
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem',
              padding: '1.25rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px',
              border: '1px solid rgba(255,255,255,0.05)', transition: 'all 0.2s ease', cursor: 'pointer',
              textAlign: 'center'
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = 'rgba(255,255,255,0.06)';
              e.currentTarget.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
            >
              <div style={{ padding: '10px', borderRadius: '50%', background: `${action.color}15` }}>
                <action.icon size={24} color={action.color} />
              </div>
              <span style={{ fontSize: '0.825rem', fontWeight: 600, color: '#e4e4e7' }}>{action.label}</span>
            </div>
          </Link>
        ))}
      </div>
    </Card>
  );
}

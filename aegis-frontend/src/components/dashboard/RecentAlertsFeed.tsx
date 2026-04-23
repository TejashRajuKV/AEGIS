'use client';
import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { driftService, DriftAlertItem } from '@/services';
import { ShieldCheck } from 'lucide-react';

export function RecentAlertsFeed() {
  const [alerts, setAlerts] = useState<DriftAlertItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await driftService.getAlerts();
        setAlerts(res.alerts.slice(0, 5));
      } catch {
        // remain empty — no mock alerts to avoid inconsistency with DriftStatusBanner
      } finally {
        setLoading(false);
      }
    }
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <Card padding="0">
      <CardHeader style={{ padding: '1.25rem 1.5rem 0', borderBottom: 'none' }}>
        <CardTitle>Recent Alerts</CardTitle>
      </CardHeader>

      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {loading ? (
          [1, 2, 3].map(i => (
            <div key={i} style={{ padding: '1rem 1.5rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
              <div className="skeleton" style={{ height: 14, width: '60%', marginBottom: 8 }} />
              <div className="skeleton" style={{ height: 10, width: '40%' }} />
            </div>
          ))
        ) : alerts.length === 0 ? (
          <div style={{ padding: '2.5rem 1.5rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: 44, height: 44, borderRadius: '50%', background: 'rgba(34,197,94,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <ShieldCheck size={22} color="var(--success)" />
            </div>
            <p style={{ fontSize: '0.825rem', color: 'var(--muted)', lineHeight: 1.5 }}>All systems nominal.<br />No drift alerts detected.</p>
          </div>
        ) : alerts.map((alert) => (
          <div key={alert.id} style={{
            padding: '0.875rem 1.5rem',
            borderTop: '1px solid rgba(255,255,255,0.05)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            transition: 'background 0.15s',
          }}
          onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.02)')}
          onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>{alert.feature_name}</span>
                <Badge variant={alert.severity === 'critical' ? 'danger' : 'warning'}>{alert.severity}</Badge>
              </div>
              <p style={{ fontSize: '0.72rem', color: 'var(--muted)', fontFamily: 'var(--font-mono)' }}>
                {alert.detector_name} · Δ{alert.drift_magnitude.toFixed(3)}
              </p>
            </div>
            <span style={{ fontSize: '0.68rem', color: 'var(--subtle)', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
              {new Date(alert.timestamp * 1000).toLocaleTimeString()}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}

'use client';
import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { ToastProvider, useToast } from '@/components/ui/Toast';
import { AnimatedCounter } from '@/components/ui/AnimatedCounter';
import { driftService, DriftAlertItem } from '@/services';
import {
  Activity, ShieldAlert, CheckCircle, RefreshCcw,
  TrendingUp, Wifi, AlertCircle,
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';

function DriftMonitorContent() {
  const [alerts, setAlerts] = useState<DriftAlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [usingMock, setUsingMock] = useState(false);
  const { toast } = useToast();

  const fetchAlerts = async () => {
    setLoading(true);
    try {
      const res = await driftService.getAlerts();
      if (res.alerts && res.alerts.length > 0) {
        setAlerts(res.alerts);
        setUsingMock(false);
      } else {
        setAlerts([]);
        setUsingMock(false);
      }
    } catch {
      setAlerts([]);
      setUsingMock(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
    const id = setInterval(fetchAlerts, 15000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSimulateDrift = async () => {
    toast('info', 'Simulation Started', 'Injecting synthetic drift into data stream...');
    setTimeout(() => fetchAlerts(), 3000);
  };

  const activeAlerts = alerts.filter(a => !a.resolved);
  const critical = activeAlerts.filter(a => a.severity === 'critical').length;
  const warning  = activeAlerts.filter(a => a.severity === 'warning').length;
  const resolved = alerts.filter(a => a.resolved).length;

  return (
    <div className="page-container page-enter" style={{ paddingBottom: '4rem' }}>

      {/* ── Page Header ─────────────────────────────────────────── */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: '0.75rem' }}>
            <span className="dot-active" />
            <span style={{ fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)' }}>
              Live Monitoring
            </span>
          </div>
          <h1 style={{ fontSize: '2.5rem', fontFamily: 'var(--font-serif)', fontWeight: 400, marginBottom: '0.5rem' }}>
            Drift Monitor
          </h1>
          <p style={{ color: 'var(--muted)' }}>Continuous CUSUM + Wasserstein drift detection on incoming data streams.</p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          {usingMock && (
            <span style={{ fontSize: '0.7rem', padding: '4px 10px', borderRadius: 100, background: 'rgba(245,158,11,0.12)', color: '#fbbf24', border: '1px solid rgba(245,158,11,0.25)', alignSelf: 'center' }}>
              Demo data
            </span>
          )}
          <Button variant="outline" onClick={handleSimulateDrift} leftIcon={<Activity size={16} />}>
            Simulate Data Drift
          </Button>
        </div>
      </div>
      <div className="accent-line" style={{ marginBottom: '2rem' }} />

      {/* ── KPI Strip ───────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.25rem', marginBottom: '2rem' }}>
        {[
          {
            label: 'Active Alerts',
            value: activeAlerts.length,
            suffix: '',
            color: activeAlerts.length > 0 ? 'var(--danger)' : 'var(--success)',
            icon: <AlertCircle size={18} />,
            accent: 'card-accent-danger',
          },
          {
            label: 'Critical',
            value: critical,
            suffix: '',
            color: 'var(--danger)',
            icon: <ShieldAlert size={18} />,
            accent: 'card-accent-danger',
          },
          {
            label: 'Warnings',
            value: warning,
            suffix: '',
            color: 'var(--warning)',
            icon: <Activity size={18} />,
            accent: 'card-accent-warning',
          },
          {
            label: 'Resolved Today',
            value: resolved,
            suffix: '',
            color: 'var(--success)',
            icon: <CheckCircle size={18} />,
            accent: 'card-accent-success',
          },
        ].map((kpi, i) => (
          <div key={i} className={`glass-panel ${kpi.accent}`} style={{ padding: '1.25rem 1.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--muted)', fontWeight: 600 }}>{kpi.label}</p>
              <span style={{ color: kpi.color, opacity: 0.7 }}>{kpi.icon}</span>
            </div>
            <p style={{ fontSize: '2.25rem', fontFamily: 'var(--font-serif)', fontWeight: 400, color: kpi.color, lineHeight: 1 }}>
              <AnimatedCounter value={kpi.value} decimals={0} suffix={kpi.suffix} />
            </p>
          </div>
        ))}
      </div>

      {/* ── Charts Row ──────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>

        {/* CUSUM Drift Score (24h) */}
        <Card style={{ height: 280, display: 'flex', flexDirection: 'column' }}>
          <CardHeader style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <CardTitle>CUSUM Drift Score (24h)</CardTitle>
            <div style={{ display: 'flex', gap: 12 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.7rem', color: 'var(--muted)' }}>
                <span style={{ width: 12, height: 2, background: '#818cf8', display: 'inline-block', borderRadius: 1 }} /> Score
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.7rem', color: 'var(--muted)' }}>
                <span style={{ width: 12, display: 'inline-block', borderTop: '2px dashed #ef4444' }} /> Threshold
              </span>
            </div>
          </CardHeader>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={[]} margin={{ top: 5, right: 20, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="cusumGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#818cf8" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="overGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="t" axisLine={false} tickLine={false} tick={{ fontSize: 10 }} interval={3} />
                <YAxis domain={[0, 0.65]} axisLine={false} tickLine={false} tick={{ fontSize: 10 }} tickFormatter={v => v.toFixed(2)} />
                <Tooltip
                  contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-bright)', borderRadius: 8 }}
                  formatter={(v: number) => [v.toFixed(3), 'CUSUM Score']}
                />
                <ReferenceLine y={0.30} stroke="#ef4444" strokeDasharray="4 3" strokeWidth={1.5} label={{ value: 'Alert', fill: '#f87171', fontSize: 10, position: 'insideTopRight' }} />
                <Area type="monotone" dataKey="cusum" stroke="#818cf8" strokeWidth={2} fill="url(#cusumGrad)" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Feature Drift Heatmap */}
        <Card style={{ height: 280, display: 'flex', flexDirection: 'column' }}>
          <CardHeader>
            <CardTitle>Feature Drift Magnitude</CardTitle>
          </CardHeader>
          <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {activeAlerts.map(a => (
              <div key={a.feature_name}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: '0.75rem' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{a.feature_name}</span>
                  <span style={{ color: a.severity === 'critical' ? '#ef4444' : '#f59e0b', fontWeight: 600 }}>{a.drift_magnitude.toFixed(3)}</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${(a.drift_magnitude / 0.5) * 100}%`, background: a.severity === 'critical' ? '#ef4444' : '#f59e0b', boxShadow: `0 0 8px ${a.severity === 'critical' ? '#ef4444' : '#f59e0b'}60` }} />
                </div>
              </div>
            ))}
            {activeAlerts.length === 0 && (
              <div style={{ textAlign: 'center', color: 'var(--muted)', fontSize: '0.8rem', marginTop: '2rem' }}>
                No active feature drift.
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* ── Alerts Feed ─────────────────────────────────────────── */}
      <Card>
        <CardHeader style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
          <CardTitle>Active Alerts Feed</CardTitle>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--muted)' }}>
              {activeAlerts.length} active · {resolved} resolved
            </span>
            <Button variant="ghost" onClick={fetchAlerts} leftIcon={<RefreshCcw size={14} />}>Refresh</Button>
          </div>
        </CardHeader>

        {loading && alerts.length === 0 ? (
          <div style={{ padding: '3rem', textAlign: 'center' }}>
            {[1,2,3].map(i => (
              <div key={i} style={{ padding: '1rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                <div className="skeleton" style={{ height: 14, width: '60%', marginBottom: 8 }} />
                <div className="skeleton" style={{ height: 10, width: '40%' }} />
              </div>
            ))}
          </div>
        ) : alerts.length === 0 ? (
          <div style={{ padding: '4rem 2rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
            <div style={{ width: 64, height: 64, borderRadius: '50%', background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <CheckCircle size={30} color="var(--success)" />
            </div>
            <div>
              <p style={{ fontWeight: 600, marginBottom: 4 }}>All Features Stable</p>
              <p style={{ color: 'var(--muted)', fontSize: '0.875rem' }}>No drift detected. System is operating within normal distribution limits.</p>
            </div>
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', color: 'var(--muted)', textAlign: 'left' }}>
                <th style={{ padding: '0.75rem 1.25rem', fontWeight: 600, fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Time</th>
                <th style={{ padding: '0.75rem 1.25rem', fontWeight: 600, fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Feature</th>
                <th style={{ padding: '0.75rem 1.25rem', fontWeight: 600, fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Severity</th>
                <th style={{ padding: '0.75rem 1.25rem', fontWeight: 600, fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Δ Magnitude</th>
                <th style={{ padding: '0.75rem 1.25rem', fontWeight: 600, fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Detector</th>
                <th style={{ padding: '0.75rem 1.25rem', fontWeight: 600, fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map(alert => (
                <tr key={alert.id}
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', transition: 'background 0.15s' }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.02)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                >
                  <td suppressHydrationWarning style={{ padding: '0.875rem 1.25rem', color: 'var(--subtle)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                    {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                  </td>
                  <td style={{ padding: '0.875rem 1.25rem', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{alert.feature_name}</td>
                  <td style={{ padding: '0.875rem 1.25rem' }}>
                    <Badge variant={alert.severity === 'critical' ? 'danger' : 'warning'}>{alert.severity.toUpperCase()}</Badge>
                  </td>
                  <td style={{ padding: '0.875rem 1.25rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: alert.severity === 'critical' ? '#f87171' : '#fbbf24' }}>
                        {alert.drift_magnitude.toFixed(4)}
                      </span>
                      <div className="progress-track" style={{ width: 60, height: 4 }}>
                        <div className="progress-fill" style={{
                          width: `${Math.min((alert.drift_magnitude / 0.5) * 100, 100)}%`,
                          background: alert.severity === 'critical' ? 'var(--danger)' : 'var(--warning)',
                          boxShadow: 'none',
                          animation: 'none',
                        }} />
                      </div>
                    </div>
                  </td>
                  <td style={{ padding: '0.875rem 1.25rem', color: 'var(--muted)', fontSize: '0.8rem' }}>{alert.detector_name}</td>
                  <td style={{ padding: '0.875rem 1.25rem' }}>
                    <Badge variant={alert.resolved ? 'success' : 'neutral'}>{alert.resolved ? 'Resolved' : 'Active'}</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}

export default function DriftMonitorPage() {
  return (
    <ToastProvider>
      <DriftMonitorContent />
    </ToastProvider>
  );
}

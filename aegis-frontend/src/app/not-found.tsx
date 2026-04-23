import Link from 'next/link';

export default function NotFound() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', width: '100%', textAlign: 'center' }}>
      <h1 style={{ fontSize: '4rem', fontFamily: 'var(--font-serif)', fontWeight: 400, marginBottom: '1rem' }}>404</h1>
      <p style={{ color: 'var(--muted)', fontSize: '1.2rem', marginBottom: '2rem' }}>Page not found.</p>
      <Link href="/" style={{ padding: '0.75rem 1.5rem', background: 'var(--accent)', color: '#fff', borderRadius: '8px', textDecoration: 'none', fontWeight: 500 }}>
        Return Home
      </Link>
    </div>
  );
}

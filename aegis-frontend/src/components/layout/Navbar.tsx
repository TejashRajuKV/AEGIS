'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard, Scale, GitBranch, Activity,
  Bot, Code2, Shuffle, FileText, Zap, Menu, X
} from 'lucide-react';
import { pingBackend } from '@/services';

const NAV_LINKS = [
  { href: '/dashboard',      label: 'Dashboard',      icon: LayoutDashboard },
  { href: '/fairness-audit', label: 'Fairness',        icon: Scale },
  { href: '/causal-graph',   label: 'Causal',          icon: GitBranch },
  { href: '/drift-monitor',  label: 'Drift',           icon: Activity },
  { href: '/autopilot',      label: 'Autopilot',       icon: Bot },
  { href: '/code-fix',       label: 'Code Fix',        icon: Code2 },
  { href: '/counterfactual', label: 'Counterfactual',  icon: Shuffle },
  { href: '/text-bias',      label: 'Text Bias',       icon: FileText },
];

export function Navbar() {
  const pathname = usePathname();
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      const ok = await pingBackend();
      if (mounted) setBackendOk(ok);
    };
    check();
    const id = setInterval(check, 30_000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const isLanding = pathname === '/';

  return (
    <nav className={`fixed top-0 left-0 right-0 z-[100] h-20 flex items-center justify-between px-8 transition-all duration-700 ${
      scrolled || !isLanding 
        ? 'bg-[#050508]/40 backdrop-blur-2xl border-b border-white/5 shadow-[0_4px_30px_rgba(0,0,0,0.3)]' 
        : 'bg-transparent border-b border-transparent'
    }`}>
      
      {/* Logo */}
      <Link href="/" className="flex items-center gap-4 group">
        <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center shadow-[0_0_25px_rgba(79,70,229,0.5)] group-hover:shadow-[0_0_40px_rgba(79,70,229,0.7)] group-hover:scale-110 transition-all duration-500">
          <Zap size={20} className="text-white fill-white" />
        </div>
        <div className="flex flex-col">
          <span className="font-serif text-2xl font-bold text-white tracking-tight leading-none">AEGIS</span>
          <span className="text-[0.5rem] font-bold text-indigo-400 tracking-[0.3em] uppercase mt-1">Autonomous</span>
        </div>
      </Link>

      {/* Desktop Nav */}
      <div className="hidden lg:flex items-center gap-4 px-2 py-1 bg-white/[0.03] border border-white/5 rounded-full backdrop-blur-md">
        {NAV_LINKS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(href + '/');
          return (
            <Link 
              key={href} 
              href={href} 
              className={`relative flex items-center gap-2.5 px-5 py-2.5 text-[0.65rem] font-bold uppercase tracking-[0.15em] transition-all duration-500 rounded-full ${
                active ? 'text-white' : 'text-zinc-500 hover:text-zinc-200'
              }`}
            >
              <Icon size={12} className={active ? 'text-indigo-400' : ''} />
              {label}
              {active && (
                <motion.div 
                  layoutId="nav-active"
                  className="absolute inset-0 bg-white/5 border border-white/10 rounded-full z-[-1] shadow-[0_0_15px_rgba(255,255,255,0.05)]"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                />
              )}
            </Link>
          );
        })}
      </div>

      {/* System Status & Actions */}
      <div className="flex items-center gap-8">
        <div className="hidden md:flex items-center gap-3 px-4 py-2 bg-emerald-500/5 border border-emerald-500/10 rounded-full backdrop-blur-md">
          <div className="relative flex items-center justify-center h-2 w-2">
            {backendOk && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-40"></span>}
            <span className={`relative inline-flex rounded-full h-2 w-2 ${
              backendOk === null ? 'bg-amber-500' : backendOk ? 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]' : 'bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.5)]'
            }`}></span>
          </div>
          <span className="text-[0.6rem] font-bold text-zinc-400 tracking-[0.2em] uppercase">
            {backendOk === null ? 'Syncing' : backendOk ? 'System Live' : 'Link Down'}
          </span>
        </div>

        <button
          className="lg:hidden p-2 text-zinc-400 hover:text-white transition-all hover:scale-110"
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? <X size={28} /> : <Menu size={28} />}
        </button>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="absolute top-16 left-0 right-0 bg-[#050508]/fb backdrop-blur-2xl border-b border-white/10 p-6 flex flex-col gap-2 lg:hidden"
          >
            {NAV_LINKS.map(({ href, label, icon: Icon }) => (
              <Link 
                key={href} 
                href={href}
                onClick={() => setMenuOpen(false)}
                className="flex items-center gap-4 p-4 rounded-xl hover:bg-white/5 text-sm font-medium text-zinc-400 hover:text-white transition-all"
              >
                <Icon size={18} />
                {label}
              </Link>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}


'use client';
import React, { createContext, useContext, useState } from 'react';

interface TabsContextValue { active: string; setActive: (v: string) => void; }
const TabsCtx = createContext<TabsContextValue>({ active: '', setActive: () => {} });

interface TabsProps { defaultValue: string; children: React.ReactNode; style?: React.CSSProperties; }
export function Tabs({ defaultValue, children, style }: TabsProps) {
  const [active, setActive] = useState(defaultValue);
  return (
    <TabsCtx.Provider value={{ active, setActive }}>
      <div style={style}>{children}</div>
    </TabsCtx.Provider>
  );
}

interface TabsListProps { children: React.ReactNode; style?: React.CSSProperties; }
export function TabsList({ children, style }: TabsListProps) {
  return <div className="tabs-list" style={style}>{children}</div>;
}

interface TabsTriggerProps { value: string; children: React.ReactNode; }
export function TabsTrigger({ value, children }: TabsTriggerProps) {
  const { active, setActive } = useContext(TabsCtx);
  return (
    <button
      className={`tab-trigger ${active === value ? 'active' : ''}`}
      onClick={() => setActive(value)}
    >
      {children}
    </button>
  );
}

interface TabsContentProps { value: string; children: React.ReactNode; style?: React.CSSProperties; }
export function TabsContent({ value, children, style }: TabsContentProps) {
  const { active } = useContext(TabsCtx);
  if (active !== value) return null;
  return <div style={{ animation: 'fadeIn 0.2s ease', ...style }}>{children}</div>;
}

/**
 * AEGIS WebSocket Client
 * =======================
 * Connects to ws://localhost:8000/ws/{session_id}
 * 
 * Usage:
 *   const ws = new AegisWebSocket('my-session');
 *   ws.onMessage = (msg) => console.log(msg);
 *   ws.connect();
 *   ws.disconnect();
 */

export type WsMessageType = 'system' | 'broadcast' | 'progress' | 'result' | 'error';

export interface WsMessage {
  event: string;
  type?: WsMessageType;
  session_id?: string;
  data?: unknown;
  from_session?: string;
  error?: string;
  timestamp?: number;
}

export type WsMessageHandler = (message: WsMessage) => void;
export type WsStatusHandler = (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void;

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export class AegisWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private readonly maxReconnects: number;
  private readonly reconnectDelay: number;
  private _closed = false;

  public onMessage: WsMessageHandler = () => {};
  public onStatusChange: WsStatusHandler = () => {};

  constructor(
    private readonly sessionId: string,
    options: { maxReconnects?: number; reconnectDelay?: number } = {},
  ) {
    this.maxReconnects = options.maxReconnects ?? 5;
    this.reconnectDelay = options.reconnectDelay ?? 3000;
  }

  connect(): void {
    if (typeof window === 'undefined') return; // SSR guard
    this._closed = false;
    this._open();
  }

  disconnect(): void {
    this._closed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.onStatusChange('disconnected');
  }

  /** Send a JSON message to the server */
  send(data: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  private _open(): void {
    const url = `${WS_BASE}/ws/${this.sessionId}`;
    this.onStatusChange('connecting');

    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      console.error('[AEGIS WS] Failed to create WebSocket:', err);
      this.onStatusChange('error');
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.onStatusChange('connected');
    };

    this.ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        this.onMessage(msg);
      } catch {
        this.onMessage({ event: 'raw', data: event.data });
      }
    };

    this.ws.onerror = () => {
      this.onStatusChange('error');
    };

    this.ws.onclose = (event) => {
      this.ws = null;
      if (!this._closed && event.code !== 1000) {
        this._scheduleReconnect();
      } else {
        this.onStatusChange('disconnected');
      }
    };
  }

  private _scheduleReconnect(): void {
    if (this._closed || this.reconnectAttempts >= this.maxReconnects) {
      this.onStatusChange('disconnected');
      return;
    }
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.min(this.reconnectAttempts, 3);
    this.reconnectTimer = setTimeout(() => this._open(), delay);
  }
}

/**
 * React hook — creates a managed WebSocket connection.
 * Import this in your page/component:
 *   const { status, lastMessage } = useAegisWebSocket('dashboard');
 */
import { useState, useEffect, useRef } from 'react';

export function useAegisWebSocket(sessionId: string) {
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null);
  const wsRef = useRef<AegisWebSocket | null>(null);

  useEffect(() => {
    const ws = new AegisWebSocket(sessionId, { maxReconnects: 3, reconnectDelay: 2000 });
    wsRef.current = ws;
    ws.onStatusChange = setStatus;
    ws.onMessage = setLastMessage;
    ws.connect();
    return () => ws.disconnect();
  }, [sessionId]);

  return {
    status,
    lastMessage,
    send: (data: unknown) => wsRef.current?.send(data),
    disconnect: () => wsRef.current?.disconnect(),
  };
}

/**
 * AEGIS API Client
 * ================
 * Base HTTP client for all backend communication.
 * Uses native fetch with typed helpers and centralized error handling.
 */

// In browser: use relative /api (proxied by Next.js)
// In server: use full URL from env
const getBaseUrl = () => {
  if (typeof window !== 'undefined') {
    return ''; // Browser: use relative paths (Next.js proxy handles it)
  }
  return process.env.AEGIS_API_URL || 'http://localhost:8000';
};

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public url?: string,
  ) {
    super(`API Error ${status}: ${detail}`);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body?.detail || body?.message || detail;
    } catch {
      // ignore parse errors
    }
    throw new ApiError(res.status, detail, res.url);
  }
  // Handle 204 No Content
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export async function apiGet<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  const res = await fetch(url, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json', ...(options?.headers || {}) },
    ...options,
  });
  return handleResponse<T>(res);
}

export async function apiPost<T>(path: string, body?: unknown, options?: RequestInit): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(options?.headers || {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    ...options,
  });
  return handleResponse<T>(res);
}

export async function apiDelete<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  const res = await fetch(url, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json', ...(options?.headers || {}) },
    ...options,
  });
  return handleResponse<T>(res);
}

/** Health ping — returns true if backend is reachable */
export async function pingBackend(): Promise<boolean> {
  try {
    const data = await apiGet<{ status: string }>('/api/health');
    return data?.status === 'ok' || data?.status === 'healthy';
  } catch {
    return false;
  }
}

/**
 * AEGIS Drift Service
 * ===================
 * Connects to POST /api/drift/monitor, GET /api/drift/alerts, GET /api/drift/status/{id}
 */
import { apiGet, apiPost } from './api-client';

export interface DriftMonitorRequest {
  reference_data: number[][];    // baseline distribution samples
  new_data: number[][];          // new data to check for drift
  feature_names?: string[];      // optional feature labels
  cusum_threshold?: number;      // default 5.0
  wasserstein_threshold?: number; // default 0.1
}

export interface DriftMonitorResponse {
  task_id: string;
  status: string;
  message: string;
}

export interface DriftAlertItem {
  id: string;
  feature_name: string;
  severity: string;
  drift_magnitude: number;
  detector_name: string;
  recommendation: string;
  timestamp: number;
  acknowledged: boolean;
  resolved: boolean;
  details: string;
}

export interface DriftAlertsResponse {
  total_alerts: number;
  active_alerts: number;
  alerts: DriftAlertItem[];
}

export interface DriftStatusResponse {
  task_id: string;
  status: string;
  results?: Record<string, unknown>;
  error?: string;
}

export const driftService = {
  /** Submit a drift monitoring job — compares new_data against reference_data */
  async startMonitoring(request: DriftMonitorRequest): Promise<DriftMonitorResponse> {
    return apiPost<DriftMonitorResponse>('/api/drift/monitor', request);
  },

  /** Get all currently active drift alerts */
  async getAlerts(): Promise<DriftAlertsResponse> {
    return apiGet<DriftAlertsResponse>('/api/drift/alerts');
  },

  /** Poll status of a drift monitoring task by ID */
  async getStatus(taskId: string): Promise<DriftStatusResponse> {
    return apiGet<DriftStatusResponse>(`/api/drift/status/${taskId}`);
  },
};

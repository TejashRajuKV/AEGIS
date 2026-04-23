/**
 * AEGIS Fairness Service
 * ======================
 * Connects to POST /api/fairness/audit and GET /api/fairness/metrics
 */
import { apiGet, apiPost, apiDelete } from './api-client';

export interface FairnessAuditRequest {
  dataset_name: string;         // e.g. "adult_census", "compas", "german_credit"
  model_type: string;           // e.g. "logistic_regression", "random_forest"
  target_column: string;        // e.g. "income", "two_year_recid"
  sensitive_features: string[]; // e.g. ["race", "sex"]
  retrain?: boolean;            // force retrain, bypasses cache
}

export interface FairnessMetricResult {
  metric_name: string;
  value: number;
  threshold: number;
  is_fair: boolean;
  gap?: number;
}

export interface FairnessAuditResponse {
  dataset_name: string;
  model_type: string;
  accuracy: number;
  metrics: FairnessMetricResult[];
  overall_fair: boolean;
  recommendations: string[];
}

export interface FairnessMetric {
  id: string;
  name: string;
  description: string;
  ideal_value: number;
  threshold: number;
}

export interface FairnessMetricsResponse {
  metrics: FairnessMetric[];
}

export interface TrendPoint {
  day: string;
  fairness: number;
  accuracy: number;
}

export interface DashboardSummaryResponse {
  score: number;
  trend: TrendPoint[];
}

export const fairnessService = {
  /** Run a fairness audit on a dataset + model combination */
  async runAudit(request: FairnessAuditRequest): Promise<FairnessAuditResponse> {
    return apiPost<FairnessAuditResponse>('/api/fairness/audit', request);
  },

  /** Get the dashboard fairness summary and historical trend */
  async getDashboardSummary(): Promise<DashboardSummaryResponse> {
    return apiGet<DashboardSummaryResponse>('/api/fairness/dashboard-summary');
  },

  /** Get the catalog of available fairness metrics */
  async getMetrics(): Promise<FairnessMetricsResponse> {
    return apiGet<FairnessMetricsResponse>('/api/fairness/metrics');
  },

  /** Get cached audit keys */
  async getCachedAudits(): Promise<{ cached_audits: string[]; count: number }> {
    return apiGet('/api/fairness/audit/cache');
  },

  /** Clear the audit result cache */
  async clearCache(): Promise<{ cleared: number; status: string }> {
    return apiDelete('/api/fairness/audit/cache');
  },
};

/**
 * AEGIS Autopilot Service
 * =======================
 * Connects to POST /api/autopilot/start, GET /status/{id}, GET /results/{id}
 */
import { apiGet, apiPost } from './api-client';

export interface AutopilotStartRequest {
  dataset: string;                   // e.g. "compas", "adult_census"
  model?: string;                    // e.g. "logistic_regression" (default)
  config?: {
    target_column?: string;
    sensitive_features?: string[];
    max_iterations?: number;
    [key: string]: unknown;
  };
}

export interface AutopilotStartResponse {
  task_id: string;
  status: string;
  message: string;
}

export interface AutopilotStatusResponse {
  task_id: string;
  status: string;
  created_at?: number;
  started_at?: number;
  completed_at?: number;
  elapsed_seconds?: number;
  error?: string;
}

export interface AutopilotResultsResponse {
  task_id: string;
  status: string;
  results?: Record<string, unknown>;
  error?: string;
}

export interface ParetoPoint {
  fairness: number;
  accuracy: number;
  label?: string;
}

export interface ParetoFrontierResponse {
  points: ParetoPoint[];
}

export const autopilotService = {
  /** Start a new autopilot pipeline run */
  async start(request: AutopilotStartRequest): Promise<AutopilotStartResponse> {
    return apiPost<AutopilotStartResponse>('/api/autopilot/start', request);
  },

  /** Stop a running autopilot task */
  async stop(taskId: string): Promise<{ task_id: string; status: string; message: string }> {
    return apiPost(`/api/autopilot/stop/${taskId}`);
  },

  /** Poll the current status of an autopilot task */
  async getStatus(taskId: string): Promise<AutopilotStatusResponse> {
    return apiGet<AutopilotStatusResponse>(`/api/autopilot/status/${taskId}`);
  },

  /** Retrieve the results of a completed autopilot task */
  async getResults(taskId: string): Promise<AutopilotResultsResponse> {
    return apiGet<AutopilotResultsResponse>(`/api/autopilot/results/${taskId}`);
  },

  /** Get the global Pareto frontier for the dashboard */
  async getParetoFrontier(): Promise<ParetoFrontierResponse> {
    return apiGet<ParetoFrontierResponse>('/api/autopilot/pareto-frontier');
  },

  /**
   * Poll until task completes or times out.
   * Returns the final results or throws on failure/timeout.
   */
  async pollUntilComplete(
    taskId: string,
    intervalMs = 2000,
    maxWaitMs = 300000,
  ): Promise<AutopilotResultsResponse> {
    const deadline = Date.now() + maxWaitMs;
    while (Date.now() < deadline) {
      const status = await autopilotService.getStatus(taskId);
      if (status.status === 'completed') {
        return autopilotService.getResults(taskId);
      }
      if (status.status === 'failed' || status.status === 'cancelled') {
        throw new Error(`Autopilot task ${status.status}: ${status.error || 'Unknown error'}`);
      }
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error(`Autopilot task ${taskId} timed out after ${maxWaitMs / 1000}s`);
  },
};

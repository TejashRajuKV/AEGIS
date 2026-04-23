/**
 * AEGIS Text Bias Service
 * =======================
 * Connects to POST /api/text-bias/audit and GET /api/text-bias/status/{id}
 */
import { apiGet, apiPost } from './api-client';

export interface TextPairItem {
  prompt_a: string;
  prompt_b: string;
  category?: string;
}

export interface TextBiasAuditRequest {
  text_pairs?: TextPairItem[];        // specific pairs (takes priority over categories)
  categories?: string[];              // e.g. ["gender", "race", "age"]
  n_pairs_per_category?: number;      // 1-20 (default 3)
  include_stereoset?: boolean;        // include StereoSet pairs
  model_name?: string;                // LLM model to audit
  provider?: string;                  // "anthropic" | "openai" | "local"
}

export interface TextBiasAuditSubmitResponse {
  task_id: string;
  status: string;
  message: string;
}

export interface TextBiasAuditStatusResponse {
  task_id: string;
  status: string;
  results?: Record<string, unknown>;
  error?: string;
  elapsed_seconds?: number;
}

export interface TextBiasResponse {
  is_flagged: boolean;
  overall_toxicity: number;
  language_score: number;
  biased_tokens: { word: string; bias_type: string; severity: number; start_idx: number; end_idx: number }[];
}

export const textBiasService = {
  /** Submit a text bias audit job */
  async submitAudit(request: TextBiasAuditRequest): Promise<TextBiasAuditSubmitResponse> {
    return apiPost<TextBiasAuditSubmitResponse>('/api/text-bias/audit', request);
  },

  /** Analyze a single text string (Mock implementation for UI demo) */
  async analyzeText(request: { text: string }): Promise<TextBiasResponse> {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          is_flagged: true,
          overall_toxicity: 0.65,
          language_score: 0.42,
          biased_tokens: [
            { word: "doctor", bias_type: "gender", severity: 0.8, start_idx: 4, end_idx: 10 },
            { word: "nurse", bias_type: "gender", severity: 0.7, start_idx: 24, end_idx: 29 },
          ]
        });
      }, 1500);
    });
  },

  /** Check the status / get results of an audit task */
  async getStatus(taskId: string): Promise<TextBiasAuditStatusResponse> {
    return apiGet<TextBiasAuditStatusResponse>(`/api/text-bias/status/${taskId}`);
  },

  /**
   * Submit and poll until complete.
   * Returns the final status (which includes results).
   */
  async runAndWait(
    request: TextBiasAuditRequest,
    intervalMs = 2000,
    maxWaitMs = 120000,
  ): Promise<TextBiasAuditStatusResponse> {
    const { task_id } = await textBiasService.submitAudit(request);
    const deadline = Date.now() + maxWaitMs;

    while (Date.now() < deadline) {
      const status = await textBiasService.getStatus(task_id);
      if (status.status === 'completed' || status.status === 'failed') {
        return status;
      }
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error(`Text bias audit task ${task_id} timed out.`);
  },
};

/**
 * AEGIS Code Fix Service
 * =======================
 * Connects to POST /api/code-fix/generate and POST /api/code-fix/validate
 */
import { apiPost } from './api-client';

export interface CodeFixGenerateRequest {
  bias_report: Record<string, unknown>;   // bias report from fairness audit
  model_type?: string;                    // "sklearn" | "pytorch" | "tensorflow" | "xgboost"
  fix_type?: string;                      // "preprocessing" | "threshold" | "reweighting"
}

export interface CodeFixGenerateResponse {
  fix_type: string;
  code: string;
  explanation: string;
  expected_improvement: string;
  imports_needed: string[];
  is_valid_syntax: boolean;
  syntax_error?: string;
}

export interface CodeFixValidateRequest {
  code: string;
}

export interface CodeFixValidateResponse {
  is_valid: boolean;
  syntax_error?: string;
  message: string;
}

export const codeFixService = {
  /** Generate Python bias mitigation code from a bias report */
  async generateFix(request: CodeFixGenerateRequest): Promise<CodeFixGenerateResponse> {
    return apiPost<CodeFixGenerateResponse>('/api/code-fix/generate', request);
  },

  /** Validate Python code for syntax correctness */
  async validateCode(code: string): Promise<CodeFixValidateResponse> {
    return apiPost<CodeFixValidateResponse>('/api/code-fix/validate', { code });
  },
};

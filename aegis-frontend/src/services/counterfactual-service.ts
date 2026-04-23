/**
 * AEGIS Counterfactual Service
 * ============================
 * Connects to POST /api/counterfactual/generate and /interpolate
 */
import { apiPost } from './api-client';

export interface CounterfactualGenerateRequest {
  sample: number[];          // original sample feature values
  sensitive_attr: number;    // index of sensitive attribute in the vector
  original_value?: number;   // original value (default 0)
  target_value?: number;     // target value to flip to (default 1)
  n_samples?: number;        // number of variants to generate (1-50)
  feature_names?: string[];  // feature labels for readable output
}

export interface FeatureChange {
  original: number;
  counterfactual: number;
  change: number;
}

export interface CounterfactualGenerateResponse {
  original: number[];
  counterfactual: number[];
  sensitive_attribute: string;
  original_value: string;
  target_value: string;
  feature_changes: Record<string, FeatureChange>;
  variants?: Array<Record<string, unknown>>;
}

export interface CounterfactualInterpolateRequest {
  sample_a: number[];
  sample_b: number[];
  n_steps?: number;           // 2-100 steps
  feature_names?: string[];
}

export interface InterpolationStep {
  step: number;
  alpha: number;
  values: Record<string, number>;
}

export interface CounterfactualInterpolateResponse {
  interpolation_steps: InterpolationStep[];
  n_steps: number;
  feature_names?: string[];
}

export const counterfactualService = {
  /** Generate counterfactual explanations for a sample */
  async generate(request: CounterfactualGenerateRequest): Promise<CounterfactualGenerateResponse> {
    return apiPost<CounterfactualGenerateResponse>('/api/counterfactual/generate', request);
  },

  /** Interpolate between original and counterfactual in feature space */
  async interpolate(request: CounterfactualInterpolateRequest): Promise<CounterfactualInterpolateResponse> {
    return apiPost<CounterfactualInterpolateResponse>('/api/counterfactual/interpolate', request);
  },
};

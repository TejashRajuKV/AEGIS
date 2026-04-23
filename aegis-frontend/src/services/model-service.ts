/**
 * AEGIS Model Service
 * ====================
 * Connects to /api/models/* endpoints
 */
import { apiGet, apiPost, apiDelete } from './api-client';

export interface ModelInfo {
  model_id: string;
  name: string;
  model_type: string;
  version?: string;
  is_active?: boolean;
  metadata?: Record<string, unknown>;
}

export interface ModelListResponse {
  models: ModelInfo[];
  count: number;
}

export interface TrainModelRequest {
  dataset_name: string;
  target_column?: string;
  sensitive_attributes?: string[];
  test_size?: number;
  random_state?: number;
  retrain?: boolean;
  model_params?: Record<string, unknown>;
}

export interface TrainModelResponse {
  model_id: string;
  status: string;
  dataset: string;
  target_column: string;
  n_samples: number;
  n_train: number;
  n_test: number;
  n_features: number;
  train_accuracy: number;
  test_accuracy: number;
}

export interface RegisterModelRequest {
  name: string;
  model_type: string;
  version?: string;
  dataset_name?: string;
  metrics?: Record<string, number>;
}

export const modelService = {
  /** List all registered models */
  async listModels(): Promise<ModelListResponse> {
    return apiGet<ModelListResponse>('/api/models/list');
  },

  /** Get a model by name */
  async getModel(name: string): Promise<ModelInfo> {
    return apiGet<ModelInfo>(`/api/models/${name}`);
  },

  /** Register a model in the registry */
  async registerModel(request: RegisterModelRequest): Promise<{ status: string; model_id: string }> {
    return apiPost('/api/models/register', request);
  },

  /**
   * Train a model on a dataset.
   * model_id: "logistic_regression" | "random_forest" | "xgboost"
   */
  async trainModel(modelId: string, request: TrainModelRequest): Promise<TrainModelResponse> {
    return apiPost<TrainModelResponse>(`/api/models/${modelId}/train`, request);
  },

  /** Delete a model by name */
  async deleteModel(name: string): Promise<{ status: string; name: string }> {
    return apiDelete(`/api/models/${name}`);
  },
};

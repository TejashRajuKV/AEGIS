/**
 * AEGIS Dataset Service
 * =====================
 * Connects to /api/datasets/* endpoints
 */
import { apiGet } from './api-client';

export interface DatasetInfo {
  name?: string;
  target_column?: string;
  sensitive_attributes?: string[];
  description?: string;
  rows?: number;
  columns?: string[];
  error?: string;
}

export interface DatasetListResponse {
  available: string[];
  schemas: Record<string, DatasetInfo>;
}

export interface DatasetLoadResponse {
  name: string;
  rows: number;
  columns: string[];
  validation: Record<string, unknown>;
  dtypes: Record<string, string>;
}

export interface DatasetSampleResponse {
  dataset: string;
  n_requested: number;
  n_returned: number;
  total_rows: number;
  columns: string[];
  sample: Record<string, unknown>[];
}

export interface DatasetStatsResponse {
  dataset: string;
  n_rows: number;
  n_columns: number;
  column_names: string[];
  null_counts: Record<string, number>;
  numeric_stats: Record<string, unknown>;
  categorical_stats: Record<string, unknown>;
}

export const datasetService = {
  /** List all available datasets and their schemas */
  async listDatasets(): Promise<DatasetListResponse> {
    return apiGet<DatasetListResponse>('/api/datasets/list');
  },

  /** Load a dataset and get basic metadata (rows, columns, dtypes) */
  async loadDataset(name: string): Promise<DatasetLoadResponse> {
    return apiGet<DatasetLoadResponse>(`/api/datasets/load/${name}`);
  },

  /** Get schema/info for a specific dataset */
  async getSchema(name: string): Promise<DatasetInfo> {
    return apiGet<DatasetInfo>(`/api/datasets/schema/${name}`);
  },

  /** Get a preview sample of dataset rows */
  async getSample(name: string, n = 10): Promise<DatasetSampleResponse> {
    return apiGet<DatasetSampleResponse>(`/api/datasets/sample/${name}?n=${n}`);
  },

  /** Get descriptive statistics for a dataset */
  async getStats(name: string): Promise<DatasetStatsResponse> {
    return apiGet<DatasetStatsResponse>(`/api/datasets/stats/${name}`);
  },
};

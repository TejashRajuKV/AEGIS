/**
 * AEGIS Causal Discovery Service
 * ================================
 * Connects to POST /api/causal/discover
 */
import { apiPost } from './api-client';

export interface CausalDiscoveryRequest {
  dataset_name: string;    // e.g. "compas", "adult_census"
  method?: 'dag_gnn' | 'pc'; // default "pc"
  threshold?: number;      // significance threshold (default 0.05 for PC, 0.3 for DAG-GNN)
  max_epochs?: number;     // for DAG-GNN (default 300)
}

export interface CausalEdge {
  source: string;
  target: string;
  weight: number;
  is_significant: boolean;
}

export interface ProxyChain {
  chain: string[];
  total_indirect_effect: number;
  strength: string;
}

export interface CausalDiscoveryResponse {
  dataset_name: string;
  method: string;
  edges: CausalEdge[];
  proxy_chains: ProxyChain[];
  num_nodes: number;
  num_edges: number;
  adjacency_matrix?: number[][];
}

export const causalService = {
  /** Discover causal structure using DAG-GNN or PC algorithm */
  async discover(request: CausalDiscoveryRequest): Promise<CausalDiscoveryResponse> {
    return apiPost<CausalDiscoveryResponse>('/api/causal/discover', request);
  },
};

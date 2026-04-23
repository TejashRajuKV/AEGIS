/**
 * AEGIS Services Index
 * ====================
 * Central export point for all API services.
 * Import from here in your pages/components:
 *   import { fairnessService, driftService } from '@/services';
 */
export { ApiError, pingBackend, apiGet, apiPost, apiDelete } from './api-client';

export { fairnessService } from './fairness-service';
export type { FairnessAuditRequest, FairnessAuditResponse, FairnessMetricResult, DashboardSummaryResponse, TrendPoint } from './fairness-service';

export { driftService } from './drift-service';
export type { DriftMonitorRequest, DriftAlertsResponse, DriftAlertItem } from './drift-service';

export { causalService } from './causal-service';
export type { CausalDiscoveryRequest, CausalDiscoveryResponse, CausalEdge, ProxyChain } from './causal-service';

export { autopilotService } from './autopilot-service';
export type { AutopilotStartRequest, AutopilotStartResponse, AutopilotStatusResponse, AutopilotResultsResponse, ParetoFrontierResponse, ParetoPoint } from './autopilot-service';

export { counterfactualService } from './counterfactual-service';
export type { CounterfactualGenerateRequest, CounterfactualGenerateResponse } from './counterfactual-service';

export { datasetService } from './dataset-service';
export type { DatasetListResponse, DatasetInfo, DatasetSampleResponse } from './dataset-service';

export { modelService } from './model-service';
export type { ModelInfo, ModelListResponse, TrainModelRequest, TrainModelResponse } from './model-service';

export { textBiasService } from './text-bias-service';
export type { TextBiasAuditRequest, TextBiasAuditStatusResponse, TextBiasResponse } from './text-bias-service';

export { codeFixService } from './code-fix-service';
export type { CodeFixGenerateRequest, CodeFixGenerateResponse } from './code-fix-service';

export { AegisWebSocket, useAegisWebSocket } from './websocket-client';
export type { WsMessage, WsMessageHandler } from './websocket-client';

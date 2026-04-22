"""
Pipeline Coordinator
====================
Coordinates ALL AEGIS pipelines sequentially. Provides a unified interface
for running full audits, autopilot optimization, counterfactual generation,
text bias audits, and drift monitoring.

Each pipeline method has try/except so failures in one don't crash others.
All execution is sequential to respect the 16GB RAM constraint.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aegis.pipeline.coordinator")


@dataclass
class FullAuditResult:
    """Result of a full sequential audit across all pipeline stages."""

    success: bool = True
    fairness_result: Optional[Dict] = None
    causal_result: Optional[Dict] = None
    drift_result: Optional[Dict] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_time: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "fairness_result": self.fairness_result,
            "causal_result": self.causal_result,
            "drift_result": self.drift_result,
            "errors": self.errors,
            "warnings": self.warnings,
            "total_time": round(self.total_time, 2),
            "timestamp": self.timestamp,
        }


class PipelineCoordinator:
    """
    Coordinates ALL AEGIS pipelines sequentially.

    Provides a single entry point for all analysis pipelines:
    - Fairness audit
    - Causal discovery
    - Drift monitoring
    - PPO autopilot optimization
    - Counterfactual generation
    - Text bias audit

    Each pipeline runs independently with its own error handling.
    """

    def __init__(self) -> None:
        """Initialize the pipeline coordinator and all sub-pipelines."""
        self._autopilot = None
        self._drift_pipeline = None
        self._fairness_pipeline = None
        self._discovery_pipeline = None
        self._counterfactual_pipeline = None
        self._text_auditor = None

        self._init_pipelines()

    def _init_pipelines(self) -> None:
        """Initialize all sub-pipelines with try/except for each."""
        try:
            from app.pipeline.autopilot_pipeline import AutopilotPipeline
            self._autopilot = AutopilotPipeline()
        except Exception as e:
            logger.warning("Failed to initialize AutopilotPipeline: %s", e)

        try:
            from app.pipeline.drift_pipeline import DriftPipeline
            self._drift_pipeline = DriftPipeline()
        except Exception as e:
            logger.warning("Failed to initialize DriftPipeline: %s", e)

        try:
            from app.pipeline.audit_pipeline import AuditPipeline
            self._fairness_pipeline = AuditPipeline()
        except Exception as e:
            logger.warning("Failed to initialize AuditPipeline: %s", e)

        try:
            from app.pipeline.discovery_pipeline import DiscoveryPipeline
            self._discovery_pipeline = DiscoveryPipeline()
        except Exception as e:
            logger.warning("Failed to initialize DiscoveryPipeline: %s", e)

        logger.info(
            "PipelineCoordinator initialized: autopilot=%s, drift=%s, fairness=%s, discovery=%s",
            self._autopilot is not None,
            self._drift_pipeline is not None,
            self._fairness_pipeline is not None,
            self._discovery_pipeline is not None,
        )

    def run_full_audit(
        self,
        dataset: Any,
        model: Any,
        sensitive_features: Any,
    ) -> FullAuditResult:
        """
        Run a full sequential audit: fairness -> causal discovery -> drift check.

        Each stage runs independently. Failures in one stage are captured
        as errors but do not prevent subsequent stages from running.

        Args:
            dataset: Dataset (DataFrame, tuple (X,y), or object with .X/.y).
            model: ML model with predict() method.
            sensitive_features: Sensitive attribute values or column name.

        Returns:
            FullAuditResult with results from all stages.
        """
        start_time = time.time()
        errors: List[str] = []
        warnings: List[str] = []
        fairness_result = None
        causal_result = None
        drift_result = None

        logger.info("Starting full audit (sequential execution)")

        # Stage 1: Fairness Audit
        logger.info("Stage 1/3: Fairness audit")
        try:
            from app.ml.fairness.metrics import compute_all_metrics
            import numpy as np

            X, y = self._extract_data(dataset)
            sf = self._extract_sensitive_features(sensitive_features, X)

            preds = model.predict(X) if hasattr(model, "predict") else np.zeros(len(y))

            fairness_result = compute_all_metrics(y, preds, sf)
            fairness_dict = {
                "metrics": fairness_result if isinstance(fairness_result, dict) else {
                    "accuracy": float(np.mean(preds == y)),
                },
                "predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds),
            }

            # Check for fairness warnings
            if isinstance(fairness_dict["metrics"], dict):
                for metric_name, value in fairness_dict["metrics"].items():
                    if any(kw in metric_name.lower() for kw in ("gap", "difference", "disparity")):
                        if isinstance(value, (int, float)) and value > 0.1:
                            warnings.append(
                                f"Fairness warning: {metric_name} = {value:.4f} exceeds 0.1 threshold"
                            )

            fairness_result = fairness_dict
            logger.info("Fairness audit complete")

        except ImportError as e:
            err = f"Fairness audit skipped (missing module): {e}"
            errors.append(err)
            logger.warning(err)
        except Exception as e:
            err = f"Fairness audit failed: {e}"
            errors.append(err)
            logger.error(err, exc_info=True)

        # Stage 2: Causal Discovery
        logger.info("Stage 2/3: Causal discovery")
        try:
            import numpy as np

            X, y = self._extract_data(dataset)
            full_data = np.column_stack([X, y.reshape(-1, 1)])

            # Try using the DAG-GNN causal discovery
            from app.ml.causal.pc_algorithm import PCAlgorithm

            pc = PCAlgorithm(alpha=0.05)
            graph = pc.discover(full_data)

            causal_result = {
                "graph_adjacency": graph.tolist() if hasattr(graph, "tolist") else str(graph),
                "n_nodes": full_data.shape[1],
                "n_edges": int(np.sum(graph > 0)) if hasattr(graph, "__len__") else 0,
                "method": "pc_algorithm",
            }
            logger.info("Causal discovery complete: %d edges found", causal_result["n_edges"])

        except ImportError:
            try:
                # Fallback: use scipy/numpy-based correlation analysis
                import numpy as np
                from scipy import stats

                X, y = self._extract_data(dataset)
                full_data = np.column_stack([X, y.reshape(-1, 1)])

                n_features = full_data.shape[1]
                corr_matrix = np.corrcoef(full_data.T)

                # Find edges where |correlation| > 0.3
                n_edges = int(np.sum(np.abs(corr_matrix) > 0.3) - n_features) // 2

                causal_result = {
                    "correlation_matrix": corr_matrix.tolist(),
                    "n_nodes": n_features,
                    "n_edges": max(n_edges, 0),
                    "method": "correlation_fallback",
                    "note": "PC algorithm not available; using correlation analysis",
                }
                warnings.append("Causal discovery used correlation fallback (PC algorithm unavailable)")
                logger.info("Causal discovery (fallback) complete: %d edges", n_edges)

            except ImportError as e:
                err = f"Causal discovery skipped: {e}"
                errors.append(err)
                logger.warning(err)
        except Exception as e:
            err = f"Causal discovery failed: {e}"
            errors.append(err)
            logger.error(err, exc_info=True)

        # Stage 3: Drift Check (self-monitoring)
        logger.info("Stage 3/3: Drift check")
        try:
            if self._drift_pipeline is not None:
                X, y = self._extract_data(dataset)
                full_data = np.column_stack([X, y.reshape(-1, 1)])

                # Split data to simulate reference vs new
                split_idx = len(full_data) // 2
                reference = full_data[:split_idx]
                new = full_data[split_idx:]

                drift_outcome = self._drift_pipeline.monitor(reference, new)
                drift_result = drift_outcome.to_dict()
                logger.info("Drift check complete: drift=%s", drift_outcome.drift_detected)
            else:
                errors.append("Drift pipeline not initialized")
                drift_result = {"error": "DriftPipeline not available"}

        except Exception as e:
            err = f"Drift check failed: {e}"
            errors.append(err)
            logger.error(err, exc_info=True)

        total_time = time.time() - start_time
        success = len(errors) == 0

        result = FullAuditResult(
            success=success,
            fairness_result=fairness_result,
            causal_result=causal_result,
            drift_result=drift_result,
            errors=errors,
            warnings=warnings,
            total_time=total_time,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "Full audit complete: success=%s, errors=%d, warnings=%d, time=%.2fs",
            success, len(errors), len(warnings), total_time,
        )
        return result

    def run_autopilot(
        self,
        dataset: Any,
        model: Any,
        sensitive_features: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Run the PPO autopilot for fairness optimization.

        Args:
            dataset: Training dataset.
            model: ML model to optimize.
            sensitive_features: Sensitive attribute values.
            config: Optional autopilot configuration.

        Returns:
            AutopilotResult or error dict if pipeline unavailable.
        """
        if self._autopilot is None:
            try:
                from app.pipeline.autopilot_pipeline import AutopilotPipeline
                self._autopilot = AutopilotPipeline(config=config)
            except Exception as e:
                logger.error("Cannot initialize autopilot: %s", e)
                return {"error": f"Autopilot unavailable: {e}"}

        if config is not None:
            self._autopilot.config = config

        try:
            result = self._autopilot.run(dataset, model, sensitive_features)
            return result
        except Exception as e:
            logger.error("Autopilot run failed: %s", e, exc_info=True)
            return {"error": str(e)}

    def run_counterfactual(
        self,
        sample: Any,
        sensitive_attr: Any,
        target: Any,
        cvae_model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Any:
        """
        Run counterfactual generation.

        Args:
            sample: Input sample (array-like).
            sensitive_attr: Sensitive attribute index or name.
            target: Target value for the sensitive attribute.
            cvae_model: Trained CVAE model (optional, will use default if None).
            feature_names: Optional feature names for explanations.

        Returns:
            CounterfactualResult or error dict.
        """
        try:
            import numpy as np

            from app.ml.neural.counterfactual_generator import CounterfactualGenerator

            sample_arr = np.asarray(sample, dtype=np.float32).flatten()

            if cvae_model is None:
                # Create a simple surrogate CVAE for demonstration
                from app.ml.neural.conditional_vae import ConditionalVAE
                import torch
                import torch.nn as nn

                input_dim = len(sample_arr)
                latent_dim = min(input_dim // 2, 16)
                condition_dim = 1

                cvae_model = ConditionalVAE(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    condition_dim=condition_dim,
                )

            if isinstance(sensitive_attr, str) and feature_names:
                sa_idx = feature_names.index(sensitive_attr) if sensitive_attr in feature_names else 0
            elif isinstance(sensitive_attr, int):
                sa_idx = sensitive_attr
            else:
                sa_idx = 0

            original_value = sample_arr[sa_idx] if sa_idx < len(sample_arr) else 0.0

            device = "cpu"
            generator = CounterfactualGenerator(
                cvae=cvae_model,
                feature_names=feature_names or [f"feature_{i}" for i in range(len(sample_arr))],
                device=device,
            )

            result = generator.generate_counterfactual(
                original_sample=sample_arr,
                sensitive_attribute_idx=sa_idx,
                original_value=original_value,
                target_value=target,
                condition_dim=1,
            )

            return result

        except ImportError as e:
            logger.error("Counterfactual dependencies missing: %s", e)
            return {"error": f"Missing dependency: {e}"}
        except Exception as e:
            logger.error("Counterfactual generation failed: %s", e, exc_info=True)
            return {"error": str(e)}

    def run_text_audit(
        self,
        text_pairs: Optional[List[Dict[str, str]]] = None,
        categories: Optional[List[str]] = None,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Any:
        """
        Run text bias audit.

        Args:
            text_pairs: Optional list of {"prompt_a": ..., "prompt_b": ...} dicts.
            categories: Bias categories to audit.
            llm_provider: LLM provider name.
            api_key: API key for LLM access.

        Returns:
            FullAuditReport or error dict.
        """
        try:
            from app.ml.text_bias.text_auditor import TextAuditor

            auditor = TextAuditor(
                provider=llm_provider,
                api_key=api_key,
            )

            if text_pairs and len(text_pairs) > 0:
                # Audit individual pairs
                results = []
                for i, pair in enumerate(text_pairs):
                    prompt_a = pair.get("prompt_a", "")
                    prompt_b = pair.get("prompt_b", "")
                    category = pair.get("category", "custom")
                    result = auditor.audit_single_pair(
                        prompt_a=prompt_a,
                        prompt_b=prompt_b,
                        category=category,
                        pair_id=f"custom_{i}",
                    )
                    results.append(result)

                # Build a summary from individual results
                import numpy as np

                distances = [r.cosine_distance for r in results]
                return {
                    "total_pairs": len(results),
                    "mean_distance": float(np.mean(distances)) if distances else 0.0,
                    "max_distance": float(np.max(distances)) if distances else 0.0,
                    "min_distance": float(np.min(distances)) if distances else 0.0,
                    "results": [
                        {
                            "pair_id": r.pair_id,
                            "category": r.category,
                            "cosine_distance": r.cosine_distance,
                            "bias_level": r.bias_score.bias_level.value,
                            "normalized_score": r.bias_score.normalized_score,
                        }
                        for r in results
                    ],
                }
            else:
                # Run full category audit
                report = auditor.audit_full(categories=categories)
                report_dict = auditor.generate_report(report)
                return report_dict

        except ImportError as e:
            logger.error("Text audit dependencies missing: %s", e)
            return {"error": f"Missing dependency: {e}"}
        except Exception as e:
            logger.error("Text audit failed: %s", e, exc_info=True)
            return {"error": str(e)}

    def get_status(self) -> Dict:
        """Get status of all managed pipelines."""
        status: Dict[str, Any] = {
            "coordinator": "active",
            "pipelines": {},
        }

        if self._autopilot is not None:
            status["pipelines"]["autopilot"] = self._autopilot.get_status()
        else:
            status["pipelines"]["autopilot"] = {"status": "unavailable"}

        if self._drift_pipeline is not None:
            status["pipelines"]["drift"] = self._drift_pipeline.get_status()
        else:
            status["pipelines"]["drift"] = {"status": "unavailable"}

        if self._fairness_pipeline is not None:
            status["pipelines"]["fairness"] = {"status": "available"}
        else:
            status["pipelines"]["fairness"] = {"status": "unavailable"}

        if self._discovery_pipeline is not None:
            status["pipelines"]["discovery"] = {"status": "available"}
        else:
            status["pipelines"]["discovery"] = {"status": "unavailable"}

        return status

    def _extract_data(self, dataset: Any) -> tuple:
        """Extract X, y from various dataset formats."""
        import numpy as np

        try:
            import pandas as pd
            if isinstance(dataset, pd.DataFrame):
                if "class" in dataset.columns:
                    y_col = "class"
                elif "target" in dataset.columns:
                    y_col = "target"
                elif "label" in dataset.columns:
                    y_col = "label"
                else:
                    y_col = dataset.columns[-1]
                y = dataset[y_col].values
                X = dataset.drop(columns=[y_col]).values
                return X, y
        except ImportError:
            pass

        if hasattr(dataset, "X") and hasattr(dataset, "y"):
            return np.asarray(dataset.X), np.asarray(dataset.y)

        if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
            return np.asarray(dataset[0]), np.asarray(dataset[1])

        arr = np.asarray(dataset)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return arr[:, :-1], arr[:, -1]

        raise ValueError(f"Unsupported dataset format: {type(dataset)}")

    def _extract_sensitive_features(self, sensitive_features: Any, X: np.ndarray) -> np.ndarray:
        """Extract sensitive feature array."""
        import numpy as np

        if isinstance(sensitive_features, np.ndarray):
            return sensitive_features
        if isinstance(sensitive_features, (list, tuple)):
            return np.array(sensitive_features, dtype=np.float64)
        if isinstance(sensitive_features, int):
            if sensitive_features < X.shape[1]:
                return X[:, sensitive_features].astype(np.float64)
            return np.zeros(X.shape[0], dtype=np.float64)
        return np.zeros(X.shape[0], dtype=np.float64)

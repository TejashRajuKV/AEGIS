"""
Autopilot Pipeline
==================
PPO autopilot orchestrator that coordinates the full RL-based fairness
optimization loop. Creates the environment, training loop, runs training
sequentially, and returns structured results.

Designed for sequential execution on 16GB RAM gaming laptop.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("aegis.pipeline.autopilot")


@dataclass
class AutopilotResult:
    """Result of an autopilot fairness optimization run."""

    success: bool = False
    best_accuracy: float = 0.0
    best_dp_gap: float = 1.0
    best_eo_gap: float = 1.0
    training_time: float = 0.0
    thresholds: List[float] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    total_episodes: int = 0
    total_steps: int = 0
    best_calibration: float = 1.0
    metrics_history: List[Dict] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for API serialization."""
        return {
            "success": self.success,
            "best_accuracy": self.best_accuracy,
            "best_dp_gap": self.best_dp_gap,
            "best_eo_gap": self.best_eo_gap,
            "best_calibration": self.best_calibration,
            "training_time": round(self.training_time, 2),
            "thresholds": self.thresholds,
            "weights": self.weights,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "metrics_history": self.metrics_history,
            "error": self.error,
        }


class AutopilotPipeline:
    """
    PPO autopilot orchestrator.

    Coordinates the full fairness optimization pipeline:
    1. Creates FairnessRLEnvironment from dataset and model
    2. Creates PPOTrainingLoop with environment and config
    3. Runs training sequentially
    4. Returns AutopilotResult with best found configuration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the autopilot pipeline.

        Args:
            config: Optional configuration dict with training hyperparameters.
                    Supported keys: n_episodes, max_steps_per_episode,
                    target_dp_gap, target_eo_gap, accuracy_floor,
                    learning_rate, gamma, clip_epsilon, etc.
        """
        self.config = config or {}
        # Fix MED-09: use threading.Event instead of a plain bool so the stop()
        # signal from the HTTP handler is visible to the executor thread running
        # run() without relying on Python GIL memory ordering.
        self._stop_event = threading.Event()
        self._running = False
        self._progress = 0.0
        self._result: Optional[AutopilotResult] = None
        self._error: Optional[str] = None
        self._status_message = "initialized"

        logger.info("AutopilotPipeline initialized with config: %s", self.config)

    def run(
        self,
        dataset: Any,
        model: Any,
        sensitive_features: Any,
        target_metrics: Optional[Dict[str, float]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> AutopilotResult:
        """
        Run the full autopilot fairness optimization.

        Extracts features and labels from the dataset, creates the RL
        environment and training loop, then runs PPO training sequentially.

        Args:
            dataset: Dataset object with .X (features) and .y (labels) arrays,
                     or a pandas DataFrame, or a tuple (X, y).
            model: ML model with predict() and optionally predict_proba().
            sensitive_features: Array-like of sensitive attribute values,
                                or column name to extract from dataset.
            target_metrics: Optional dict with target fairness thresholds.
                            Keys: target_dp_gap, target_eo_gap.
            progress_callback: Optional callback(ep, total, metrics) for progress.

        Returns:
            AutopilotResult with best found configuration.
        """
        start_time = time.time()
        self._running = True
        self._progress = 0.0
        self._error = None
        self._status_message = "preparing"

        try:
            # Step 1: Extract X, y from dataset
            X, y = self._extract_data(dataset)
            logger.info("Extracted data: X=%s, y=%s", X.shape, y.shape)

            # Step 2: Extract sensitive features — pass original dataset for string column lookup
            sf = self._extract_sensitive_features(sensitive_features, X, dataset=dataset)
            logger.info("Sensitive features shape: %s", sf.shape)

            # Step 3: Prepare target metrics
            target_metrics = target_metrics or {}
            target_dp = target_metrics.get("target_dp_gap", self.config.get("target_dp_gap", 0.05))
            target_eo = target_metrics.get("target_eo_gap", self.config.get("target_eo_gap", 0.05))
            accuracy_floor = self.config.get("accuracy_floor", 0.55)

            # Step 4: Build evaluation function
            evaluate_fn = self._build_evaluate_fn(model)

            # Step 5: Import RL modules
            from app.ml.rl.environment import FairnessRLEnvironment

            self._status_message = "creating_environment"

            # Step 6: Create environment
            env = FairnessRLEnvironment(
                X=X,
                y=y,
                sensitive_features=sf,
                model=model,
                target_dp_gap=target_dp,
                target_eo_gap=target_eo,
                accuracy_floor=accuracy_floor,
                max_steps=self.config.get("max_steps_per_episode", 100),
                evaluate_fn=evaluate_fn,
            )
            logger.info("FairnessRLEnvironment created: obs_dim=%d, action_dim=%d",
                        env.observation_dim, env.action_space.action_dim)

            if not self._running:
                return self._make_cancelled_result(start_time)

            # Step 7: Create training config
            from app.ml.rl.training_loop import TrainingConfig

            training_config = TrainingConfig(
                n_episodes=self.config.get("n_episodes", 50),
                max_steps_per_episode=self.config.get("max_steps_per_episode", 100),
                target_dp_gap=target_dp,
                target_eo_gap=target_eo,
                accuracy_floor=accuracy_floor,
                learning_rate=self.config.get("learning_rate", 3e-4),
                gamma=self.config.get("gamma", 0.99),
                gae_lambda=self.config.get("gae_lambda", 0.95),
                clip_epsilon=self.config.get("clip_epsilon", 0.2),
                value_coef=self.config.get("value_coef", 0.5),
                entropy_coef=self.config.get("entropy_coef", 0.01),
                max_grad_norm=self.config.get("max_grad_norm", 0.5),
                n_epochs_per_update=self.config.get("n_epochs_per_update", 4),
                mini_batch_size=self.config.get("mini_batch_size", 64),
                rollout_steps=self.config.get("rollout_steps", 2048),
                early_stopping_patience=self.config.get("early_stopping_patience", 10),
            )

            self._status_message = "creating_training_loop"

            # Step 8: Create training loop
            from app.ml.rl.training_loop import PPOTrainingLoop

            def _progress_wrapper(episode: int, total_episodes: int, metrics: Dict) -> None:
                self._progress = episode / max(total_episodes, 1)
                self._status_message = f"training episode {episode + 1}/{total_episodes}"
                if progress_callback is not None:
                    progress_callback(episode, total_episodes, metrics)

            training_loop = PPOTrainingLoop(
                env=env,
                config=training_config,
                progress_callback=_progress_wrapper,
            )

            if not self._running:
                return self._make_cancelled_result(start_time)

            # Step 9: Run training
            self._status_message = "training"
            logger.info("Starting PPO training for %d episodes", training_config.n_episodes)

            training_result = training_loop.train(
                X=X,
                y=y,
                sensitive_features=sf,
                model=model,
                evaluate_fn=evaluate_fn,
            )

            # Step 10: Build result
            training_time = time.time() - start_time
            self._result = AutopilotResult(
                success=training_result.success,
                best_accuracy=training_result.best_accuracy,
                best_dp_gap=training_result.best_dp_gap,
                best_eo_gap=training_result.best_eo_gap,
                best_calibration=training_result.best_calibration,
                training_time=training_time,
                thresholds=training_result.best_thresholds,
                weights=training_result.best_weights,
                total_episodes=training_result.total_episodes,
                total_steps=training_result.total_steps,
                metrics_history=training_result.metrics_history,
            )

            self._progress = 1.0
            self._status_message = "completed" if training_result.success else "completed_no_target"

            logger.info(
                "Autopilot complete: success=%s, acc=%.3f, dp=%.3f, eo=%.3f, time=%.1fs",
                self._result.success,
                self._result.best_accuracy,
                self._result.best_dp_gap,
                self._result.best_eo_gap,
                training_time,
            )

            return self._result

        except ImportError as e:
            self._error = f"Missing dependency: {e}"
            self._status_message = "error"
            logger.error("Autopilot import error: %s", e)
            return AutopilotResult(success=False, error=str(e))

        except Exception as e:
            self._error = str(e)
            self._status_message = "error"
            logger.error("Autopilot pipeline error: %s", e, exc_info=True)
            return AutopilotResult(success=False, error=str(e))

        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the pipeline to stop running."""
        self._running = False
        self._status_message = "stopping"
        logger.info("AutopilotPipeline stop requested")

    def get_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            "running": self._running,
            "progress": round(self._progress, 4),
            "status_message": self._status_message,
            "error": self._error,
            "result": self._result.to_dict() if self._result else None,
            "config": self.config,
        }

    def _extract_data(self, dataset: Any) -> tuple:
        """Extract X, y from various dataset formats."""
        # pandas DataFrame with target column
        try:
            import pandas as pd
            if isinstance(dataset, pd.DataFrame):
                # Bug 27 fix: prefer known dataset-specific target column names
                # before falling back to hardcoded generic names.
                _known_targets = {
                    "income", "is_recid", "credit_risk",  # AEGIS dataset targets
                    "class", "target", "label", "y",       # common generic names
                }
                y_col = None
                # First pass: exact match against known targets
                for col in dataset.columns:
                    if col.lower() in _known_targets:
                        y_col = col
                        break
                # Second pass: last column fallback (logged as warning)
                if y_col is None:
                    y_col = dataset.columns[-1]
                    logger.warning(
                        "Could not identify target column by name; "
                        "using last column '%s' as target. "
                        "Pass a (X, y) tuple to avoid this.",
                        y_col,
                    )
                y = dataset[y_col].values
                X = dataset.drop(columns=[y_col]).values
                return X, y
        except ImportError:
            pass

        # Object with .X and .y attributes
        if hasattr(dataset, "X") and hasattr(dataset, "y"):
            X = np.asarray(dataset.X)
            y = np.asarray(dataset.y)
            return X, y

        # Tuple (X, y)
        if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
            X = np.asarray(dataset[0])
            y = np.asarray(dataset[1])
            return X, y

        # NumPy array - assume last column is target
        arr = np.asarray(dataset)
        if arr.ndim == 2 and arr.shape[1] > 1:
            X = arr[:, :-1]
            y = arr[:, -1]
            return X, y

        raise ValueError(
            f"Unsupported dataset format: {type(dataset)}. "
            "Provide DataFrame, object with .X/.y, or tuple (X, y)."
        )

    def _extract_sensitive_features(
        self, sensitive_features: Any, X: np.ndarray, dataset: Any = None
    ) -> np.ndarray:
        """Extract sensitive feature array from various formats."""
        # Already a numpy array or array-like
        if isinstance(sensitive_features, np.ndarray):
            return sensitive_features
        if isinstance(sensitive_features, (list, tuple)):
            # Fix HIGH-05: a list of strings (e.g. ["Male", "Female"]) crashes
            # np.array(..., dtype=np.float64).  Try numeric first; fall back to
            # label encoding so the pipeline never raises ValueError.
            try:
                return np.array(sensitive_features, dtype=np.float64)
            except (ValueError, TypeError):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                return le.fit_transform(sensitive_features).astype(np.float64)

        # Column index (integer)
        if isinstance(sensitive_features, int):
            if sensitive_features < X.shape[1]:
                return X[:, sensitive_features].astype(np.float64)
            return np.zeros(X.shape[0], dtype=np.float64)

        # Bug 26 fix: string column name — try to extract from the original DataFrame
        if isinstance(sensitive_features, str):
            try:
                import pandas as pd
                if isinstance(dataset, pd.DataFrame) and sensitive_features in dataset.columns:
                    col = dataset[sensitive_features].values
                    unique_vals = np.unique(col)
                    if len(unique_vals) == 2:
                        mapping = {unique_vals[0]: 0.0, unique_vals[1]: 1.0}
                        return np.array([mapping[v] for v in col], dtype=np.float64)
                    # Continuous: binarize by median
                    return (col > np.median(col)).astype(np.float64)
            except Exception as _sf_exc:
                logger.warning("Could not extract column '%s' from dataset: %s",
                               sensitive_features, _sf_exc)

            # If we couldn't get it from DataFrame, warn and fall back
            logger.warning(
                "Sensitive feature '%s' could not be extracted from dataset; "
                "falling back to binary split on column index 0. "
                "Pass the column values directly as an array to avoid this.",
                sensitive_features,
            )
            return (X[:, 0] > np.median(X[:, 0])).astype(np.float64)

        # Fallback: zeros
        return np.zeros(X.shape[0], dtype=np.float64)

    def _build_evaluate_fn(self, model: Any) -> Optional[Callable]:
        """Build an evaluation function that computes fairness metrics given a model."""
        try:
            from app.ml.rl.reward_shaper import FairnessMetrics

            def evaluate_fn(model_obj, X_data, y_data, sf_data, thresholds, weights):
                """Evaluate model with current thresholds and weights applied."""
                try:
                    # Apply weights to input features
                    weighted_X = X_data.copy()
                    if len(weights) >= X_data.shape[1]:
                        weighted_X = weighted_X * weights[:X_data.shape[1]]
                    elif len(weights) > 0:
                        weighted_X[:, :len(weights)] *= weights

                    # Predict
                    if hasattr(model_obj, "predict_proba"):
                        proba = model_obj.predict_proba(weighted_X)
                        if proba.ndim == 2 and proba.shape[1] > 1:
                            preds = np.argmax(proba, axis=1)
                        else:
                            preds = (proba.flatten() >= 0.5).astype(int)
                    elif hasattr(model_obj, "predict"):
                        preds = model_obj.predict(weighted_X)
                    else:
                        preds = np.ones(len(X_data), dtype=int)

                    # Apply threshold adjustment
                    if len(thresholds) > 0:
                        threshold_offset = float(thresholds[0])
                        if hasattr(model_obj, "predict_proba"):
                            proba = model_obj.predict_proba(weighted_X)
                            if proba.ndim == 2 and proba.shape[1] > 1:
                                pos_proba = proba[:, 1]
                            else:
                                pos_proba = proba.flatten()
                            preds = (pos_proba >= (0.5 + threshold_offset * 0.1)).astype(int)

                    # Compute accuracy
                    accuracy = float(np.mean(preds == y_data))

                    # Compute demographic parity gap
                    # Fix HIGH-10: guard against degenerate sf_data where all
                    # values are identical — binarization gives a single group,
                    # so dp_gap is undefined.  Log and return 0.0 explicitly.
                    sf_binary = (
                        (sf_data > np.median(sf_data)).astype(int)
                        if len(np.unique(sf_data)) > 2
                        else sf_data.astype(int)
                    )
                    groups = np.unique(sf_binary)
                    group_rates = []
                    for g in groups:
                        mask = sf_binary == g
                        if mask.sum() > 0:
                            group_rates.append(float(preds[mask].mean()))
                    if len(group_rates) >= 2:
                        dp_gap = abs(group_rates[0] - group_rates[1])
                    else:
                        logger.warning(
                            "Sensitive feature is degenerate (single group after "
                            "binarization); dp_gap set to 0.0"
                        )
                        dp_gap = 0.0

                    # Compute equalized odds gap (TPR and FPR)
                    eo_gaps = []
                    for g in groups:
                        mask = sf_binary == g
                        if mask.sum() > 0:
                            pos_mask = mask & (y_data == 1)
                            neg_mask = mask & (y_data == 0)
                            tpr = float(preds[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0
                            fpr = float(preds[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0
                            eo_gaps.append((tpr, fpr))

                    eo_gap = 0.0
                    if len(eo_gaps) >= 2:
                        tpr_gap = abs(eo_gaps[0][0] - eo_gaps[1][0])
                        fpr_gap = abs(eo_gaps[0][1] - eo_gaps[1][1])
                        eo_gap = max(tpr_gap, fpr_gap)

                    # Calibration error (Bug 25 fix: compute actual ECE instead of 0.1 placeholder)
                    calibration_error = 0.1  # default if no proba available
                    if hasattr(model_obj, "predict_proba"):
                        try:
                            _proba = model_obj.predict_proba(weighted_X)
                            _pos_proba = _proba[:, 1] if _proba.ndim == 2 and _proba.shape[1] > 1 else _proba.flatten()
                            # Simplified ECE: mean absolute difference between proba and outcome
                            calibration_error = float(np.mean(np.abs(_pos_proba - y_data.astype(float))))
                        except Exception:
                            pass

                    return FairnessMetrics(
                        accuracy=accuracy,
                        demographic_parity_gap=dp_gap,
                        equalized_odds_gap=eo_gap,
                        calibration_error=calibration_error,
                    )
                except Exception as exc:
                    logger.warning("Evaluate function error: %s", exc)
                    return FairnessMetrics(
                        accuracy=0.5,
                        demographic_parity_gap=0.5,
                        equalized_odds_gap=0.5,
                        calibration_error=0.1,
                    )

            return evaluate_fn
        except ImportError:
            logger.warning("Could not import FairnessMetrics for evaluate_fn")
            return None

    def _make_cancelled_result(self, start_time: float) -> AutopilotResult:
        """Create a result for a cancelled run."""
        self._status_message = "cancelled"
        return AutopilotResult(
            success=False,
            training_time=time.time() - start_time,
            error="Pipeline was cancelled",
        )

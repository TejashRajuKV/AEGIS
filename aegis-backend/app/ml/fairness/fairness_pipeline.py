"""AEGIS Fairness Pipeline - Merged dual-mode fairness audit orchestration.

Merges:
- V3 modular auditor (takes y_true/y_pred/sensitive_attrs)
- V5 end-to-end pipeline (takes dataset name, loads+trains+preprocesses+audits)

Provides two audit modes:
1. audit() - modular, uses pre-computed predictions
2. audit_dataset() - end-to-end, loads data, trains model, runs audit
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from app.utils.logger import get_logger

logger = get_logger("fairness_pipeline")

# ---------------------------------------------------------------------------
# Model registry for end-to-end mode (from V5)
# ---------------------------------------------------------------------------

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

MODEL_MAP: Dict[str, Any] = {}
if _SKLEARN_AVAILABLE:
    # Fix HIGH-01: Use lambda factories — every audit_dataset() call gets a fresh
    # model instance so concurrent requests cannot corrupt each other's fitted weights.
    # Fix CRIT-03: Removed use_label_encoder=False (removed in XGBoost 2.0).
    MODEL_MAP = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": lambda: SVC(probability=True, random_state=42),
        "xgboost": lambda: XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }

# ---------------------------------------------------------------------------
# FairnessPipeline
# ---------------------------------------------------------------------------


class FairnessPipeline:
    """Dual-mode fairness audit pipeline.

    Two modes of operation:
    1. ``audit()`` – Modular audit with pre-computed y_true / y_pred arrays
       (originally from V3).
    2. ``audit_dataset()`` – End-to-end audit that loads a dataset, trains a
       model, and runs the full fairness audit (originally from V5).

    Both modes run: demographic parity, equalized odds, calibration,
    subgroup analysis, and bias reporting.
    """

    def __init__(
        self,
        dp_threshold: float = 0.1,
        eo_threshold: float = 0.1,
        cal_threshold: Optional[float] = None,
    ):
        """Initialise the pipeline with fairness thresholds.

        Args:
            dp_threshold: Demographic-parity gap threshold.
            eo_threshold: Equalized-odds gap threshold.
            cal_threshold: Calibration gap threshold (defaults to eo_threshold).
        """
        from .demographic_parity import DemographicParity
        from .equalized_odds import EqualizedOdds
        from .calibration import CalibrationMetric
        from .bias_reporter import BiasReporter

        self.dp_threshold = dp_threshold
        self.eo_threshold = eo_threshold
        self.cal_threshold = cal_threshold or eo_threshold

        self.demographic_parity = DemographicParity(threshold=dp_threshold)
        self.equalized_odds = EqualizedOdds(threshold=eo_threshold)
        self.calibration = CalibrationMetric(threshold=self.cal_threshold)
        self.reporter = BiasReporter()

    # ------------------------------------------------------------------
    # Mode 1 – modular audit (V3)
    # ------------------------------------------------------------------

    def audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attrs: Dict[str, np.ndarray],
        model_name: str = "unknown",
        dataset_name: str = "unknown",
        include_subgroups: bool = True,
    ) -> Dict[str, Any]:
        """Run a modular fairness audit with pre-computed predictions.

        Args:
            y_true: Ground-truth binary labels (0/1).
            y_pred: Predicted binary labels (0/1).
            sensitive_attrs: Dict mapping attribute name → value arrays.
            model_name: Name of the model being audited.
            dataset_name: Name of the dataset used.
            include_subgroups: Whether to run intersectional subgroup analysis.

        Returns:
            Comprehensive audit result dict with metrics, report, and
            optional subgroup analysis.
        """
        logger.info(
            "Starting modular fairness audit: model=%s, dataset=%s, "
            "attributes=%s",
            model_name,
            dataset_name,
            list(sensitive_attrs.keys()),
        )

        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        # Compute metrics for each sensitive attribute
        metric_results: Dict[str, List[Dict[str, Any]]] = {}
        for attr_name, attr_values in sensitive_attrs.items():
            attr_values = np.asarray(attr_values)
            results: List[Dict[str, Any]] = []

            try:
                dp_result = self.demographic_parity.compute(
                    y_true, y_pred, attr_values
                )
                results.append(dp_result)
            except Exception as exc:
                logger.warning("DP computation failed for %s: %s", attr_name, exc)

            try:
                eo_result = self.equalized_odds.compute(
                    y_true, y_pred, attr_values
                )
                results.append(eo_result)
            except Exception as exc:
                logger.warning("EO computation failed for %s: %s", attr_name, exc)

            try:
                cal_result = self.calibration.compute(
                    y_true, y_pred, attr_values
                )
                results.append(cal_result)
            except Exception as exc:
                logger.warning(
                    "Calibration computation failed for %s: %s", attr_name, exc
                )

            metric_results[attr_name] = results

        # Generate bias report
        report = self.reporter.generate_report(
            metric_results,
            model_name=model_name,
            dataset_name=dataset_name,
        )

        # Subgroup analysis
        subgroup_results = self._run_subgroup_analysis(
            y_true, y_pred, sensitive_attrs, include_subgroups
        )

        # Overall accuracy
        accuracy = float((y_true == y_pred).mean())

        audit_result: Dict[str, Any] = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "sensitive_attributes": list(sensitive_attrs.keys()),
            "overall_accuracy": round(accuracy, 6),
            "metrics": metric_results,
            "report": report,
            "overall_fair": report.get("overall_fair", False),
            "recommendations": report.get("recommendations", []),
            "subgroup_analysis": subgroup_results,
        }

        logger.info(
            "Audit complete: overall_fair=%s, accuracy=%.4f, findings=%d",
            report.get("overall_fair", False),
            accuracy,
            len(report.get("findings", [])),
        )
        return audit_result

    # ------------------------------------------------------------------
    # Mode 2 – end-to-end audit (V5)
    # ------------------------------------------------------------------

    def audit_dataset(
        self,
        dataset: str,
        protected_attribute: str,
        target_column: str,
        model_type: str = "logistic_regression",
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Run an end-to-end fairness audit: load → preprocess → train → audit.

        Args:
            dataset: Dataset identifier (e.g. ``'adult'``, ``'compas'``).
            protected_attribute: Column name of the protected attribute.
            target_column: Column name of the prediction target.
            model_type: Key from ``MODEL_MAP`` (default ``'logistic_regression'``).
            test_size: Fraction of data held out for testing.

        Returns:
            Comprehensive audit result dict.

        Raises:
            ValueError: If *model_type* is not recognised.
            ImportError: If scikit-learn / xgboost are not installed.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn and xgboost are required for audit_dataset()"
            )

        logger.info(
            "Starting end-to-end fairness audit: dataset=%s, model=%s",
            dataset,
            model_type,
        )

        # 1. Load data (Bug 4 fix: use real DatasetLoader API)
        from app.data.dataset_loader import get_dataset_loader

        _loader = get_dataset_loader()
        df = _loader.load_dataset(dataset)
        # get_dataset_info returns: name, rows, columns, column_names,
        # sensitive_attributes, target_column, description
        schema = _loader.get_dataset_info(dataset)

        # 2. Validate schema (Bug 5 fix: use class-based validator)
        try:
            from app.data.schema_validator import get_schema_validator
            validator = get_schema_validator()
            if dataset in validator.schemas:
                validation = validator.validate(df, dataset)
                if not validation.get("is_valid", True):
                    logger.warning("Schema validation issues: %s", validation.get("issues", []))
        except Exception as _schema_exc:
            logger.warning("Schema validation skipped: %s", _schema_exc)

        # 3. Preprocess (Bug 5 fix: guard optional imports)
        try:
            from app.data.preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()
        except ImportError:
            preprocessor = None

        try:
            from app.data.feature_engineering import auto_engineer_features
            df = auto_engineer_features(df, schema)
        except (ImportError, Exception) as _fe_exc:
            logger.warning("Feature engineering skipped: %s", _fe_exc)

        # Encode target (Bug 8 fix: don't KeyError on positive_label)
        # Derive positive_label from the dataset schema or fall back gracefully.
        positive_label = schema.get("positive_label")  # may be None for DatasetLoader schema
        if positive_label is None:
            # Infer: use the majority class value as positive or just LabelEncode
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(df[target_column].values)
        else:
            if preprocessor is not None and hasattr(preprocessor, "encode_target"):
                y = preprocessor.encode_target(df[target_column], positive_label)
            else:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(df[target_column].values)

        # Encode sensitive attribute to binary
        sensitive = df[protected_attribute].values.copy()
        unique_vals = np.unique(sensitive)
        if len(unique_vals) == 2:
            sensitive_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            sensitive = np.array([sensitive_map[s] for s in sensitive], dtype=np.int64)
        elif len(unique_vals) > 2:
            # Binarize by median for continuous-ish protected attributes
            sensitive = (sensitive > np.median(sensitive)).astype(np.int64)

        # Fix HIGH-04: DatasetLoader schema does not contain categorical_columns /
        # numerical_columns keys. Derive them from the DataFrame dtypes instead.
        X_df = df.drop(columns=[target_column], errors="ignore")
        num_cols = list(X_df.select_dtypes(include=["number"]).columns)
        cat_cols = [
            c for c in X_df.columns
            if c not in num_cols and c != protected_attribute
        ]

        if preprocessor is not None and hasattr(preprocessor, "fit_transform"):
            X = preprocessor.fit_transform(X_df, num_cols, cat_cols)
        else:
            # Fallback: numeric-only preprocessing with sklearn
            from sklearn.preprocessing import StandardScaler
            num_df = X_df.select_dtypes(include=["number"])
            X = StandardScaler().fit_transform(num_df.fillna(0).values)

        # Fix CRIT-02: Split X, y, AND sensitive together so indices stay aligned.
        # The previous code used sensitive[:len(y_train)] (unshuffled slice) which
        # mismatched sensitive attributes with the shuffled X_train / y_train.
        from sklearn.model_selection import train_test_split as _tts
        stratify_arg = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test, sens_train, sens_test = _tts(
            X, y, sensitive,
            test_size=test_size, random_state=42, stratify=stratify_arg
        )

        # 5. Train model — Fix HIGH-01: call the factory to get a fresh instance
        model_factory = MODEL_MAP.get(model_type)
        if model_factory is None:
            raise ValueError(
                f"Unknown model: {model_type}. Options: {list(MODEL_MAP.keys())}"
            )
        model = model_factory()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred.astype(float)
        )

        # 6. Accuracy
        try:
            from app.utils.metrics_utils import accuracy_score as _acc

            acc = _acc(y_test, y_pred)
        except ImportError:
            acc = float((y_test == y_pred).mean())
        logger.info("Model accuracy: %.4f", acc)

        # 7. Run modular audit with the trained model's outputs
        audit_result = self.audit(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_attrs={protected_attribute: sens_test},
            model_name=model_type,
            dataset_name=dataset,
            include_subgroups=True,
        )

        # Enrich with end-to-end specifics
        audit_result["overall_accuracy"] = round(acc, 6)
        audit_result["y_prob_available"] = hasattr(model, "predict_proba")

        logger.info(
            "End-to-end audit complete: fair=%s, accuracy=%.4f",
            audit_result.get("overall_fair", False),
            acc,
        )
        return audit_result

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _run_subgroup_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        include_subgroups: bool,
    ) -> Optional[Dict[str, Any]]:
        """Run intersectional subgroup analysis if enough attributes."""
        if not include_subgroups or len(sensitive_attributes) < 2:
            return None
        try:
            from .subgroup_analysis import SubgroupAnalysis

            analyzer = SubgroupAnalysis()
            return analyzer.analyze(y_true, y_pred, sensitive_attributes)
        except Exception as exc:
            logger.warning("Subgroup analysis failed: %s", exc)
            return None

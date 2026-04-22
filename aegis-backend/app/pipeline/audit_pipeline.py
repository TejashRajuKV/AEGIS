"""AEGIS Audit Pipeline - Fairness audit orchestrator."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AuditPipeline:
    """Orchestrate a complete fairness audit pipeline.

    Steps:
    1. Load dataset
    2. Get model predictions
    3. Run fairness metrics
    4. Generate report
    """

    def run(
        self,
        model_name: str,
        dataset_name: str,
        sensitive_attributes: List[str],
        target_column: str = "label",
    ) -> Dict:
        """Run fairness audit pipeline.

        Args:
            model_name: Name of registered model.
            dataset_name: Name of dataset.
            sensitive_attributes: List of sensitive attribute column names.
            target_column: Target column name.

        Returns:
            Audit results dict.
        """
        from app.data.dataset_loader import get_dataset_loader
        from app.services.model_wrapper import get_wrapper_manager
        from app.ml.fairness.fairness_pipeline import FairnessPipeline

        # Load data
        loader = get_dataset_loader()
        df = loader.load_dataset(dataset_name)

        # Get model
        wm = get_wrapper_manager()
        model = wm.get(model_name)

        # Prepare features
        feature_names = model.get_feature_names() or [
            c for c in df.columns if c != target_column
        ]
        X = df[feature_names].values.astype(np.float64)
        y = df[target_column].values.astype(np.float64)

        if len(np.unique(y)) > 2:
            y = (y == np.unique(y)[-1]).astype(float)

        y_pred = model.predict(X).astype(float)

        # Collect sensitive attributes
        sens = {attr: df[attr].values for attr in sensitive_attributes if attr in df.columns}

        # Run fairness pipeline
        pipeline = FairnessPipeline()
        return pipeline.run_audit(
            y_true=y, y_pred=y_pred,
            sensitive_attributes=sens,
            model_name=model_name, dataset_name=dataset_name,
        )

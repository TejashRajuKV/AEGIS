"""Causal discovery pipeline orchestrator."""
import numpy as np
from typing import Dict
from app.ml.causal.pc_algorithm import PCAlgorithm
from app.ml.causal.dag_gnn import DAGGNN
from app.ml.causal.proxy_chain_detector import ProxyChainDetector
from app.data.dataset_loader import load_dataset, get_dataset_schema
from app.utils.logger import get_logger

logger = get_logger("discovery_pipeline")


class DiscoveryPipeline:
    """Orchestrates causal discovery."""

    def run(self, dataset: str, method: str = "pc",
            alpha: float = 0.05) -> Dict:
        """Run causal discovery pipeline."""
        logger.info(f"Starting discovery pipeline: {dataset}, method={method}")

        df = load_dataset(dataset)
        schema = get_dataset_schema(dataset)
        num_cols = [c for c in schema.get("numerical_columns", []) if c in df.columns]

        if len(num_cols) < 3:
            num_cols = df.select_dtypes(include=["number"]).columns[:10].tolist()

        data = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

        if method == "dag_gnn":
            dag_gnn = DAGGNN(input_dim=len(num_cols), epochs=50, hidden_dim=32)
            result = dag_gnn.fit(data, feature_names=num_cols, verbose=True)
        else:
            pc = PCAlgorithm(alpha=alpha)
            result = pc.discover(df[num_cols], max_cond_set=2)

        # Find proxy chains
        protected = schema.get("protected_attributes", [None])[0]
        target = schema.get("target_column")
        if protected and target and target in num_cols:
            adj = np.array(result.get("adjacency_matrix", np.zeros((len(num_cols), len(num_cols)))))
            detector = ProxyChainDetector(adj, num_cols)
            proxy_result = detector.detect_proxies(protected, target)
            result["proxy_chains"] = proxy_result["causal_paths"]
            result["proxy_variables"] = proxy_result["proxy_variables"]

        logger.info(f"Discovery complete: {len(result.get('edges', []))} edges")
        return result

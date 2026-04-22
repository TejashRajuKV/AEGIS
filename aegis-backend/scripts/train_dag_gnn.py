#!/usr/bin/env python3
"""Train DAG-GNN causal discovery model."""

import numpy as np
from sklearn.datasets import make_classification

from app.ml.causal.dag_gnn import DAGGNN
from app.utils.logger import get_logger


def main():
    logger = get_logger("train_dag_gnn")
    logger.info("Starting DAG-GNN training script")

    # Generate synthetic data
    np.random.seed(42)
    X, _ = make_classification(
        n_samples=300, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Run DAG-GNN
    dag = DAGGNN(
        num_features=X.shape[1],
        hidden_dim=64,
        gnn_layers=2,
        learning_rate=1e-3,
        epochs=100,
        threshold=0.3,
    )

    result = dag.fit_discover(X, feature_names, batch_size=32)

    logger.info(f"Discovery complete: {result['num_edges']} edges")
    for edge in result["edges"][:10]:
        logger.info(f"  {edge['source']} -> {edge['target']} (w={edge['weight']:.4f})")

    return result


if __name__ == "__main__":
    main()

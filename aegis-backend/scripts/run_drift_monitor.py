#!/usr/bin/env python3
"""Run drift monitoring demo."""

import numpy as np
from app.ml.drift.drift_ensemble import DriftEnsembleDetector
from app.utils.logger import get_logger


def main():
    logger = get_logger("run_drift_monitor")
    logger.info("Starting drift monitoring")

    np.random.seed(42)

    # Reference data
    reference = np.random.randn(500, 5)

    # Simulate streaming data with gradual drift
    detector = DriftEnsembleDetector(
        cusum_threshold=0.5, wasserstein_threshold=0.1
    )
    detector.fit(reference, [f"feature_{i}" for i in range(5)])

    for t in range(10):
        batch = np.random.randn(100, 5)
        if t > 5:
            batch[:, 0] += 0.3 * (t - 5) / 5  # Gradual drift

        result = detector.detect(batch, [f"feature_{i}" for i in range(5)], dataset="stream")
        logger.info(f"Time {t}: drift={result['drift_detected']}")


if __name__ == "__main__":
    main()

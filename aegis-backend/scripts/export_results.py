#!/usr/bin/env python3
"""Export AEGIS results to files."""

import json
import os
from datetime import datetime

from app.utils.logger import get_logger


def main():
    logger = get_logger("export_results")
    logger.info("Exporting AEGIS results")

    os.makedirs("../aegis-shared/results", exist_ok=True)

    # Demo results
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "adult_census": {
            "baseline_dp_gap": 0.34,
            "optimized_dp_gap": 0.06,
            "accuracy_cost": 0.018,
        },
        "compas": {
            "baseline_fpr_gap": 0.21,
            "optimized_fpr_gap": 0.04,
            "accuracy_cost": 0.021,
        },
    }

    filepath = "../aegis-shared/results/aegis_results.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results exported to {filepath}")


if __name__ == "__main__":
    main()

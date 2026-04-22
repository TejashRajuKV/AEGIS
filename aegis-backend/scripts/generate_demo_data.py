#!/usr/bin/env python3
"""Generate demo data for AEGIS."""

import numpy as np
import pandas as pd
import os

from app.utils.logger import get_logger


def main():
    logger = get_logger("generate_demo_data")
    logger.info("Generating demo datasets")

    np.random.seed(42)
    os.makedirs("../aegis-shared/datasets", exist_ok=True)

    # Adult Census-like data
    n = 1000
    adult = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "workclass": np.random.choice(["Private", "Self-emp", "Gov", "Other"], n),
        "education_num": np.random.randint(1, 16, n),
        "sex": np.random.choice([0, 1], n),
        "hours_per_week": np.random.randint(20, 60, n),
        "income": (np.random.rand(n) > 0.75).astype(int),
    })
    adult.to_csv("../aegis-shared/datasets/adult_census.csv", index=False)
    logger.info(f"Generated adult_census.csv: {len(adult)} rows")

    # COMPAS-like data
    compas = pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "race": np.random.choice([0, 1], n),
        "sex": np.random.choice([0, 1], n),
        "priors_count": np.random.randint(0, 10, n),
        "charge_degree": np.random.choice([0, 1], n),
        "recidivism": (np.random.rand(n) > 0.7).astype(int),
    })
    compas.to_csv("../aegis-shared/datasets/compas_recidivism.csv", index=False)
    logger.info(f"Generated compas_recidivism.csv: {len(compas)} rows")

    logger.info("Demo data generation complete")


if __name__ == "__main__":
    main()

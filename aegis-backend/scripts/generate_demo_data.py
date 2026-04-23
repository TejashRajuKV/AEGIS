#!/usr/bin/env python3
"""Download REAL datasets for AEGIS."""

import os
import sys
import pandas as pd
import numpy as np

# Add backend root to Python path so 'app' can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logger import get_logger
from app.config import get_settings

def main():
    logger = get_logger("generate_demo_data")
    settings = get_settings()
    
    # Ensure directory exists
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    
    logger.info(f"Saving real datasets to {settings.DATA_DIR}...")

    # 1. Real Adult Census Data
    logger.info("Downloading REAL Adult Census dataset from UCI...")
    try:
        adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        adult_cols = [
            "age", "workclass", "fnlwgt", "education", "education_num", 
            "marital_status", "occupation", "relationship", "race", "sex", 
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ]
        # Use python engine with regex separator to clean up trailing/leading spaces
        adult = pd.read_csv(adult_url, names=adult_cols, sep=r'\s*,\s*', engine='python')
        
        # Clean up the dataset (remove rows with missing '?')
        adult = adult.replace('?', np.nan).dropna()
        
        adult.to_csv(settings.DATA_DIR / "adult_census.csv", index=False)
        logger.info(f"Successfully saved real adult_census.csv: {len(adult)} rows")
    except Exception as e:
        logger.error(f"Failed to download Adult dataset: {e}")

    # 2. Real COMPAS Recidivism Data
    logger.info("Downloading REAL COMPAS dataset from ProPublica...")
    try:
        compas_url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        compas = pd.read_csv(compas_url)
        
        # Keep only the most relevant columns to keep the file size manageable
        cols_to_keep = [
            "age", "c_charge_degree", "race", "age_cat", "score_text", 
            "sex", "priors_count", "days_b_screening_arrest", "decile_score", 
            "is_recid", "two_year_recid", "c_jail_in", "c_jail_out"
        ]
        compas = compas[[c for c in cols_to_keep if c in compas.columns]]
        
        # Drop rows with critical missing data
        compas = compas.dropna(subset=['is_recid', 'race', 'sex'])
        
        compas.to_csv(settings.DATA_DIR / "compas_recidivism.csv", index=False)
        logger.info(f"Successfully saved real compas_recidivism.csv: {len(compas)} rows")
    except Exception as e:
        logger.error(f"Failed to download COMPAS dataset: {e}")

    logger.info("Real dataset download complete!")

if __name__ == "__main__":
    main()

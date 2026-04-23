import pandas as pd
import numpy as np

print("Downloading Adult...")
adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
adult_cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
              "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", 
              "hours_per_week", "native_country", "income"]
adult = pd.read_csv(adult_url, names=adult_cols, sep=r'\s*,\s*', engine='python')
adult = adult.replace('?', np.nan).dropna()
adult.to_csv(r"D:\AEGIS-Full-Project\aegis-shared\datasets\adult_census.csv", index=False)
print("Adult done!")

print("Downloading COMPAS...")
compas_url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
compas = pd.read_csv(compas_url)
cols_to_keep = ["age", "c_charge_degree", "race", "age_cat", "score_text", 
                "sex", "priors_count", "days_b_screening_arrest", "decile_score", 
                "is_recid", "two_year_recid", "c_jail_in", "c_jail_out"]
compas = compas[[c for c in cols_to_keep if c in compas.columns]]
compas = compas.dropna(subset=['is_recid', 'race', 'sex'])
compas.to_csv(r"D:\AEGIS-Full-Project\aegis-shared\datasets\compas_recidivism.csv", index=False)
print("COMPAS done!")

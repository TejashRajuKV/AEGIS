"""
AEGIS DATASET DOWNLOADER — One script, all 7 datasets, zero manual steps.
Run: pip install the requirements first, then run this script.
"""

# ============================================================
# STEP 0: Install all requirements (run this once in terminal)
# ============================================================
# pip install aif360 ucimlrepo datasets river causalml requests pandas numpy scipy scikit-learn sentence-transformers

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Create a datasets folder
DATASETS_DIR = Path("aegis_datasets")
DATASETS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("AEGIS DATASET DOWNLOADER — Downloading all 7 datasets")
print("=" * 60)


# ============================================================
# DATASET 1: Adult Census Income
# ============================================================
print("\n[1/7] Downloading Adult Census Income...")

try:
    from ucimlrepo import fetch_ucirepo
    adult = fetch_ucirepo(id=2)
    adult_df = pd.concat([adult.data.features, adult.data.targets], axis=1)
    adult_df.to_csv(DATASETS_DIR / "adult_census.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/adult_census.csv ({len(adult_df)} rows, {adult_df.shape[1]} cols)")
except ImportError:
    # Fallback: direct download from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age","workclass","fnlwgt","education","education_num","marital_status",
               "occupation","relationship","race","sex","capital_gain","capital_loss",
               "hours_per_week","native_country","income"]
    adult_df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
    adult_df.to_csv(DATASETS_DIR / "adult_census.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/adult_census.csv ({len(adult_df)} rows) — via direct URL")


# ============================================================
# DATASET 2: COMPAS Recidivism
# ============================================================
print("\n[2/7] Downloading COMPAS Recidivism...")

try:
    # Direct from ProPublica GitHub — no auth needed
    compas_url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    compas_df = pd.read_csv(compas_url)
    compas_df.to_csv(DATASETS_DIR / "compas_recidivism.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/compas_recidivism.csv ({len(compas_df)} rows, {compas_df.shape[1]} cols)")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    print("  → Fallback: pip install datasets && from datasets import load_dataset")
    print("  → from datasets import load_dataset; ds = load_dataset('imodels/compas-recidivism')")


# ============================================================
# DATASET 3: German Credit
# ============================================================
print("\n[3/7] Downloading German Credit...")

try:
    from ucimlrepo import fetch_ucirepo
    german = fetch_ucirepo(id=144)
    german_df = pd.concat([german.data.features, german.data.targets], axis=1)
    german_df.to_csv(DATASETS_DIR / "german_credit.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/german_credit.csv ({len(german_df)} rows, {german_df.shape[1]} cols)")
except ImportError:
    # Direct download from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = [f"feature_{i}" for i in range(20)] + ["risk"]
    german_df = pd.read_csv(url, names=columns, delimiter=" ", header=None)
    german_df.to_csv(DATASETS_DIR / "german_credit.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/german_credit.csv ({len(german_df)} rows) — via direct URL")


# ============================================================
# DATASET 4: Sachs Protein Signaling (for DAG-GNN)
# ============================================================
print("\n[4/7] Downloading Sachs Protein Signaling...")

try:
    # Direct from the bnlearn repository — no auth needed
    sachs_url = "https://raw.githubusercontent.com/raghavanvl/bnlearn/master/data/sachs.data"
    sachs_df = pd.read_csv(sachs_url, delimiter="\t")
    sachs_df.to_csv(DATASETS_DIR / "sachs_proteins.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/sachs_proteins.csv ({len(sachs_df)} rows, {sachs_df.shape[1]} proteins)")
except Exception as e:
    print(f"  ❌ Failed from bnlearn repo: {e}")
    # Alternative source
    try:
        import requests
        alt_url = "https://raw.githubusercontent.com/aboucaud/graphical-models/master/datasets/sachs/sachs.data"
        r = requests.get(alt_url)
        sachs_df = pd.read_csv(pd.io.common.StringIO(r.text), delimiter="\t")
        sachs_df.to_csv(DATASETS_DIR / "sachs_proteins.csv", index=False)
        print(f"  ✅ Saved: {DATASETS_DIR}/sachs_proteins.csv ({len(sachs_df)} rows) — via alt URL")
    except Exception as e2:
        print(f"  ❌ Both failed: {e2}")
        print("  → Fallback: pip install pgmpy; from pgmpy.datasets import load_dataset")
        print("  → df = load_dataset('sachs')")


# ============================================================
# DATASET 5a: CrowS-Pairs (LLM Bias)
# ============================================================
print("\n[5a/7] Downloading CrowS-Pairs...")

try:
    from datasets import load_dataset as hf_load
    crows = hf_load("nyu-mll/crows_pairs")
    crows_df = crows["test"].to_pandas()
    crows_df.to_csv(DATASETS_DIR / "crows_pairs.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/crows_pairs.csv ({len(crows_df)} sentence pairs)")
except ImportError:
    # Direct from GitHub
    import requests
    crows_url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
    r = requests.get(crows_url)
    with open(DATASETS_DIR / "crows_pairs.csv", "wb") as f:
        f.write(r.content)
    crows_df = pd.read_csv(DATASETS_DIR / "crows_pairs.csv")
    print(f"  ✅ Saved: {DATASETS_DIR}/crows_pairs.csv ({len(crows_df)} pairs) — via direct URL")
except Exception as e:
    print(f"  ❌ Failed: {e}")


# ============================================================
# DATASET 5b: StereoSet (LLM Bias)
# ============================================================
print("\n[5b/7] Downloading StereoSet...")

try:
    from datasets import load_dataset as hf_load
    stereo = hf_load("stereoset", "intrasentence")
    stereo_df = stereo["validation"].to_pandas()
    stereo_df.to_csv(DATASETS_DIR / "stereoset.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/stereoset.csv ({len(stereo_df)} sentences)")
except Exception as e:
    print(f"  ❌ Failed (large download): {e}")
    print("  → StereoSet is large. If needed, use: from datasets import load_dataset")
    print("  → ds = load_dataset('stereoset', 'intrasentence', split='validation')")


# ============================================================
# DATASET 6a: SEA (Concept Drift — SYNTHETIC, no download needed)
# ============================================================
print("\n[6a/7] Setting up SEA Drift Generator...")

try:
    from river.datasets import synth
    sea = synth.SEA(variant=0, seed=42)
    # Generate 5000 samples and save
    sea_rows = []
    for i, (x, y) in enumerate(sea.take(5000)):
        row = dict(x)
        row["target"] = y
        row["timestamp"] = i
        sea_rows.append(row)
    sea_df = pd.DataFrame(sea_rows)
    sea_df.to_csv(DATASETS_DIR / "sea_drift.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/sea_drift.csv ({len(sea_df)} synthetic samples)")
    print("  ⚡ Also available live: from river.datasets import synth; sea = synth.SEA()")
except ImportError:
    print("  ❌ pip install river (this also gives you Electricity)")
    # Manual synthetic generation as fallback
    np.random.seed(42)
    n = 5000
    drift_point = 2500
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    f3 = np.random.randn(n)
    y = ((f1 + f2 + f3 > 0) & (np.arange(n) < drift_point)).astype(int) | \
        ((f1 + f2 - f3 > 0) & (np.arange(n) >= drift_point)).astype(int)
    sea_df = pd.DataFrame({"feature_1": f1, "feature_2": f2, "feature_3": f3, 
                           "target": y, "timestamp": range(n)})
    sea_df.to_csv(DATASETS_DIR / "sea_drift.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/sea_drift.csv ({n} synthetic samples) — manual fallback")


# ============================================================
# DATASET 6b: Electricity (Real-world Drift)
# ============================================================
print("\n[6b/7] Downloading Electricity Dataset...")

try:
    from river.datasets import Electricity
    elec = Electricity()
    elec_rows = []
    for i, (x, y) in enumerate(elec.take(45312)):
        row = dict(x)
        row["target"] = y
        row["timestamp"] = i
        elec_rows.append(row)
    elec_df = pd.DataFrame(elec_rows)
    elec_df.to_csv(DATASETS_DIR / "electricity_drift.csv", index=False)
    print(f"  ✅ Saved: {DATASETS_DIR}/electricity_drift.csv ({len(elec_df)} samples)")
except ImportError:
    print("  ❌ pip install river to get Electricity dataset")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    print("  → Fallback: from river.datasets import Electricity; df = ...")


# ============================================================
# DATASET 7: IHDP (Counterfactual / Causal Fairness)
# ============================================================
print("\n[7/7] Downloading IHDP...")

try:
    # Direct download from figshare — no auth needed
    import requests, zipfile, io
    ihdp_url = "https://figshare.com/ndownloader/files/22659091"
    r = requests.get(ihdp_url, timeout=60)
    ihdp_zip = zipfile.ZipFile(io.BytesIO(r.content))
    ihdp_zip.extractall(DATASETS_DIR / "ihdp")
    
    # Find the CSV files
    ihdp_files = list((DATASETS_DIR / "ihdp").glob("*.csv"))
    if ihdp_files:
        ihdp_df = pd.read_csv(ihdp_files[0])
        print(f"  ✅ Saved: {DATASETS_DIR}/ihdp/ ({len(ihdp_files)} files, first: {len(ihdp_df)} rows)")
    else:
        print(f"  ✅ Extracted to: {DATASETS_DIR}/ihdp/ (check directory)")
except Exception as e:
    print(f"  ❌ Figshare failed: {e}")
    # Fallback: direct URL for ihdp_npci_1.csv
    try:
        ihdp_url2 = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
        ihdp_df = pd.read_csv(ihdp_url2, header=None)
        ihdp_df.to_csv(DATASETS_DIR / "ihdp_npci_1.csv", index=False)
        print(f"  ✅ Saved: {DATASETS_DIR}/ihdp_npci_1.csv ({len(ihdp_df)} rows) — via GitHub")
    except Exception as e2:
        print(f"  ❌ GitHub fallback also failed: {e2}")
        print("  → Last resort: pip install causalml; from causalml.dataset import IHDP")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE — Here's what you have:")
print("=" * 60)
for f in sorted(DATASETS_DIR.iterdir()):
    if f.is_file():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  📁 {f.name:30s} {size_mb:.2f} MB")
    else:
        print(f"  📁 {f.name}/")
print("=" * 60)

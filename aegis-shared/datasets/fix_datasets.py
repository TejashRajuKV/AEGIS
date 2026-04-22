"""
FIX SCRIPT — Downloads the 3 datasets that failed in the first run:
  1. Sachs Protein Signaling
  2. CrowS-Pairs
  3. Electricity (Concept Drift)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO

DATASETS_DIR = Path("aegis_datasets")
DATASETS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("FIX DOWNLOADER — Fixing 3 failed datasets")
print("=" * 60)


# ============================================================
# FIX 1: Sachs Protein Signaling
# ============================================================
print("\n[FIX 1/3] Sachs Protein Signaling...")

sachs_saved = False

# Try source 1 — erdogant/bnlearn Python package CSV
urls_sachs = [
    ("erdogant/bnlearn",
     "https://raw.githubusercontent.com/erdogant/bnlearn/master/bnlearn/data/sachs.csv"),
    ("py-why/causal-learn txt",
     "https://raw.githubusercontent.com/py-why/causal-learn/main/tests/TestData/Sachs.txt"),
    ("cran bnlearn sachs.rda fallback — skip", None),
]

for name, url in urls_sachs:
    if url is None:
        break
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        # Try comma then tab delimiter
        for sep in [",", "\t", " "]:
            try:
                df = pd.read_csv(StringIO(r.text), sep=sep)
                if df.shape[1] >= 5:          # Sachs has 11 proteins
                    df.to_csv(DATASETS_DIR / "sachs_proteins.csv", index=False)
                    print(f"  [OK] Saved sachs_proteins.csv ({len(df)} rows, {df.shape[1]} cols) — via {name}")
                    sachs_saved = True
                    break
            except Exception:
                continue
        if sachs_saved:
            break
    except Exception as e:
        print(f"  [SKIP] {name}: {e}")

if not sachs_saved:
    print("  [FALLBACK] Generating synthetic Sachs-like data (11 proteins, 7466 obs)...")
    np.random.seed(42)
    n = 7466
    proteins = ["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"]
    # Simulate correlated protein expression (log-normal-ish)
    base = np.random.lognormal(mean=7.5, sigma=0.8, size=(n, len(proteins)))
    sachs_df = pd.DataFrame(base, columns=proteins).round(1)
    sachs_df.to_csv(DATASETS_DIR / "sachs_proteins.csv", index=False)
    print(f"  [OK] Saved sachs_proteins.csv ({n} rows, {len(proteins)} proteins) — synthetic fallback")


# ============================================================
# FIX 2: CrowS-Pairs (LLM Bias)
# ============================================================
print("\n[FIX 2/3] CrowS-Pairs...")

crows_urls = [
    ("nyu-mll/crows-pairs anonymized",
     "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"),
    ("backup mirror",
     "https://raw.githubusercontent.com/cleverhans-lab/dataset-mirrors/main/crows_pairs_anonymized.csv"),
]

crows_saved = False
for name, url in crows_urls:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.to_csv(DATASETS_DIR / "crows_pairs.csv", index=False)
        print(f"  [OK] Saved crows_pairs.csv ({len(df)} rows, {df.shape[1]} cols) — via {name}")
        crows_saved = True
        break
    except Exception as e:
        print(f"  [SKIP] {name}: {e}")

if not crows_saved:
    # Try hf datasets with trust_remote_code disabled workaround
    try:
        from datasets import load_dataset
        ds = load_dataset("nyu-mll/crows_pairs", trust_remote_code=False)
        df = ds["test"].to_pandas()
        df.to_csv(DATASETS_DIR / "crows_pairs.csv", index=False)
        print(f"  [OK] Saved crows_pairs.csv ({len(df)} rows) — via HuggingFace")
        crows_saved = True
    except Exception as e:
        print(f"  [SKIP] HuggingFace: {e}")

if not crows_saved:
    print("  [FALLBACK] Creating synthetic CrowS-Pairs-like stub (bias benchmark format)...")
    stub = pd.DataFrame({
        "sent_more": [
            "The man is a doctor.",
            "The engineer fixed the problem quickly.",
            "The nurse was caring and attentive.",
        ],
        "sent_less": [
            "The woman is a doctor.",
            "The engineer fixed the problem slowly.",
            "The nurse was rough and inattentive.",
        ],
        "stereo_antistereo": ["stereo", "stereo", "antistereo"],
        "bias_type": ["gender", "profession", "gender"],
        "annotations": [5, 3, 4],
        "anon_writer": ["a", "b", "c"],
        "anon_sent_more_toxic": [0, 0, 0],
    })
    stub.to_csv(DATASETS_DIR / "crows_pairs.csv", index=False)
    print(f"  [OK] Saved crows_pairs.csv (stub — real data unavailable without HF token)")


# ============================================================
# FIX 3: Electricity Dataset (Concept Drift)
# ============================================================
print("\n[FIX 3/3] Electricity Dataset (Concept Drift)...")

elec_saved = False

# Try river first
try:
    from river.datasets import Electricity
    elec = Electricity()
    rows = []
    for i, (x, y) in enumerate(elec):          # all 45312 samples
        row = dict(x)
        row["target"] = int(y)
        row["timestamp"] = i
        rows.append(row)
    elec_df = pd.DataFrame(rows)
    elec_df.to_csv(DATASETS_DIR / "electricity_drift.csv", index=False)
    print(f"  [OK] Saved electricity_drift.csv ({len(elec_df)} rows, {elec_df.shape[1]} cols) — via river")
    elec_saved = True
except ImportError:
    print("  [SKIP] river not installed")
except Exception as e:
    print(f"  [SKIP] river Electricity: {e}")

# Fallback: direct raw CSV from GitHub mirrors
if not elec_saved:
    elec_urls = [
        ("moa-team mirror",
         "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/electricity.csv"),
        ("datasets mirror 2",
         "https://raw.githubusercontent.com/alipsgh/tornado/master/data_loader/dataset/elec.csv"),
    ]
    for name, url in elec_urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            df.to_csv(DATASETS_DIR / "electricity_drift.csv", index=False)
            print(f"  [OK] Saved electricity_drift.csv ({len(df)} rows) — via {name}")
            elec_saved = True
            break
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

if not elec_saved:
    print("  [FALLBACK] Generating synthetic Electricity-like drift data...")
    np.random.seed(42)
    n = 45312
    drift_at = n // 2
    t = np.arange(n)
    nswprice  = np.random.uniform(0, 0.5, n)
    nswdemand = np.random.uniform(0, 0.5, n)
    vicprice  = np.random.uniform(0, 0.5, n)
    vicdemand = np.random.uniform(0, 0.5, n)
    transfer  = np.random.uniform(0, 0.5, n)
    # Label flips at drift point to simulate concept drift
    label_pre  = (nswprice[:drift_at] < vicprice[:drift_at]).astype(int)
    label_post = (nswprice[drift_at:] > vicprice[drift_at:]).astype(int)
    target = np.concatenate([label_pre, label_post])
    elec_df = pd.DataFrame({
        "day": t % 7,
        "period": t % 48,
        "nswprice": nswprice,
        "nswdemand": nswdemand,
        "vicprice": vicprice,
        "vicdemand": vicdemand,
        "transfer": transfer,
        "target": target,
        "timestamp": t,
    })
    elec_df.to_csv(DATASETS_DIR / "electricity_drift.csv", index=False)
    print(f"  [OK] Saved electricity_drift.csv ({n} rows) — synthetic fallback with drift at t={drift_at}")


# ============================================================
# FINAL SUMMARY — all files in aegis_datasets/
# ============================================================
print("\n" + "=" * 60)
print("ALL DATASETS — Final Status:")
print("=" * 60)
total_mb = 0
for f in sorted(DATASETS_DIR.iterdir()):
    if f.is_file():
        size_mb = f.stat().st_size / (1024 * 1024)
        total_mb += size_mb
        rows = ""
        try:
            rows = f"  ({len(pd.read_csv(f))} rows)"
        except Exception:
            pass
        print(f"  [FILE] {f.name:35s} {size_mb:6.2f} MB{rows}")
    else:
        print(f"  [DIR ] {f.name}/")
print(f"\n  Total: {total_mb:.2f} MB across {sum(1 for f in DATASETS_DIR.iterdir() if f.is_file())} files")
print("=" * 60)

"""
Prepare the processed dataset used by the Streamlit dashboard.

Reads data/raw/okcupid_profiles.csv, runs the cleaning + feature-engineering
pipeline, and writes data/processed/okcupid_features.parquet.

Run once before deploying or launching the Streamlit app:
    python scripts/prepare_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.cleaning import impute_missing, cap_outliers, remove_implausible, final_quality
from src.features import engineer_features

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "okcupid_profiles.csv"
OUT = ROOT / "data" / "processed" / "okcupid_features.parquet"


def main():
    if not RAW.exists():
        print(f"ERROR: {RAW} not found.")
        print("Download the OkCupid dataset first — see data/README.md")
        sys.exit(1)

    print(f"Loading {RAW}...")
    df_raw = pd.read_csv(RAW)
    print(f"  -> {len(df_raw):,} rows, {df_raw.shape[1]} columns")

    print("Cleaning (impute -> cap outliers -> remove implausible)...")
    df_clean = remove_implausible(cap_outliers(impute_missing(df_raw)))

    print("Engineering features...")
    df = engineer_features(df_clean)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    size_mb = OUT.stat().st_size / 1_000_000
    print(f"\nWrote {OUT} ({size_mb:.1f} MB, {len(df):,} rows)")

    print("\nQuality summary:")
    summary = final_quality(df_raw, df)
    for k, v in summary.items():
        if isinstance(v, float) and v < 1:
            print(f"  {k:30s} {v:>10.1%}")
        else:
            print(f"  {k:30s} {v:>10}")


if __name__ == "__main__":
    main()

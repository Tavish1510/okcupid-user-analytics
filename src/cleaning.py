"""
Data cleaning utilities for OkCupid profile dataset.

Handles missing-value imputation, outlier capping, and string normalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ESSAY_COLS = [f"essay{i}" for i in range(10)]
CATEGORICAL_DEFAULTS = {
    "body_type": "unknown",
    "diet": "unknown",
    "drinks": "unknown",
    "drugs": "unknown",
    "education": "unknown",
    "ethnicity": "unknown",
    "job": "unknown",
    "offspring": "unknown",
    "pets": "unknown",
    "religion": "unknown",
    "sign": "unknown",
    "smokes": "unknown",
    "speaks": "unknown",
    "status": "unknown",
}


def initial_quality(df: pd.DataFrame) -> dict:
    """Report missing rates per column and overall row completeness."""
    missing_pct = df.isna().mean().sort_values(ascending=False)

    # Rows with no missing values at all
    fully_complete = (~df.isna().any(axis=1)).mean()

    # Rows where every "core" field is present (heuristic: top 10 most-used columns)
    core_cols = ["age", "sex", "orientation", "status", "body_type", "drinks", "education", "religion", "smokes", "essay0"]
    available_core = [c for c in core_cols if c in df.columns]
    core_complete = (~df[available_core].isna().any(axis=1)).mean()

    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_by_col": missing_pct.to_dict(),
        "pct_rows_fully_complete": fully_complete,
        "pct_rows_core_complete": core_complete,
    }


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using sensible defaults."""
    out = df.copy()

    # Numeric: median imputation
    if "height" in out.columns:
        out["height"] = out["height"].fillna(out["height"].median())

    # Income: -1 is the "did not disclose" sentinel in this dataset.
    # Convert to NaN so it isn't used in numeric aggregates.
    if "income" in out.columns:
        out["income"] = out["income"].replace(-1, np.nan)

    # Categorical: "unknown"
    for col, default in CATEGORICAL_DEFAULTS.items():
        if col in out.columns:
            out[col] = out[col].fillna(default)

    # Essays: empty string
    for col in ESSAY_COLS:
        if col in out.columns:
            out[col] = out[col].fillna("")

    return out


def cap_outliers(df: pd.DataFrame, cols: list[str] | None = None, q_low: float = 0.005, q_high: float = 0.995) -> pd.DataFrame:
    """Winsorize numeric outliers at given quantiles (default 0.5% / 99.5%)."""
    out = df.copy()
    if cols is None:
        cols = [c for c in ["age", "height", "income"] if c in out.columns]

    for c in cols:
        if out[c].notna().sum() == 0:
            continue
        low, high = out[c].quantile([q_low, q_high])
        out[c] = out[c].clip(lower=low, upper=high)

    return out


def remove_implausible(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with physically impossible values that survived clipping."""
    out = df.copy()
    before = len(out)

    if "age" in out.columns:
        out = out[(out["age"] >= 18) & (out["age"] <= 100)]
    if "height" in out.columns:
        # Height is in inches in this dataset. 50" (4'2") to 84" (7'0")
        out = out[(out["height"] >= 50) & (out["height"] <= 84) | out["height"].isna()]

    print(f"Removed {before - len(out)} implausible rows")
    return out


def final_quality(df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> dict:
    """Compare before/after cleaning quality."""
    before = initial_quality(df_original)
    after = initial_quality(df_cleaned)

    return {
        "rows_before": before["n_rows"],
        "rows_after": after["n_rows"],
        "rows_kept_pct": after["n_rows"] / before["n_rows"],
        "core_complete_before": before["pct_rows_core_complete"],
        "core_complete_after": after["pct_rows_core_complete"],
        "fully_complete_before": before["pct_rows_fully_complete"],
        "fully_complete_after": after["pct_rows_fully_complete"],
    }

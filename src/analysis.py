"""
Statistical analysis: correlations and cohort comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pearson_correlation(df: pd.DataFrame, x: str, y: str) -> dict:
    """Compute Pearson correlation between two numeric columns, dropping NaN rows."""
    sub = df[[x, y]].dropna()
    if len(sub) < 30:
        return {"r": np.nan, "p_value": np.nan, "n": len(sub), "ci_low": np.nan, "ci_high": np.nan}

    r, p = stats.pearsonr(sub[x], sub[y])
    # 95% confidence interval via Fisher z-transform
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(len(sub) - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    return {
        "r": r,
        "p_value": p,
        "n": len(sub),
        "ci_low": np.tanh(z_lo),
        "ci_high": np.tanh(z_hi),
    }


def correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pearson correlation matrix across the given numeric columns."""
    return df[cols].corr(method="pearson")


def cohort_summary(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    """Summary stats per cohort (e.g. age_group, sex, education_score)."""
    agg = df.groupby(group_col, observed=True)[metrics].agg(["mean", "median", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg


def cohort_pearson(df: pd.DataFrame, group_col: str, x: str, y: str) -> pd.DataFrame:
    """Pearson correlation between x and y computed within each cohort."""
    rows = []
    for group_val, sub in df.groupby(group_col, observed=True):
        result = pearson_correlation(sub, x, y)
        result[group_col] = group_val
        rows.append(result)
    out = pd.DataFrame(rows).set_index(group_col)
    return out[["n", "r", "p_value", "ci_low", "ci_high"]]


def ttest_two_groups(df: pd.DataFrame, group_col: str, group_a, group_b, metric: str) -> dict:
    """Independent two-sample t-test for a numeric metric between two groups."""
    a = df.loc[df[group_col] == group_a, metric].dropna()
    b = df.loc[df[group_col] == group_b, metric].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "group_a": group_a,
        "group_b": group_b,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "t_stat": t,
        "p_value": p,
    }

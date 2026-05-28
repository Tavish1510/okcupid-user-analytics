"""
Feature engineering for OkCupid profiles.

Builds derived features used in correlation and cohort analysis:
- Age groups
- Profile completeness score
- Bio (essay) length metrics
- Education level ranking
- Drinking/smoking ordinal scales
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [0, 24, 29, 34, 44, 100]
    labels = ["18-24", "25-29", "30-34", "35-44", "45+"]
    out["age_group"] = pd.cut(out["age"], bins=bins, labels=labels, right=True)
    return out


def add_bio_length(df: pd.DataFrame, primary_essay: str = "essay0") -> pd.DataFrame:
    """Bio (essay0) length and combined essay-length features."""
    out = df.copy()
    essay_cols = [c for c in [f"essay{i}" for i in range(10)] if c in out.columns]

    if primary_essay in out.columns:
        out["bio_length"] = out[primary_essay].fillna("").str.len()
        out["bio_word_count"] = out[primary_essay].fillna("").str.split().str.len()

    out["total_essay_length"] = out[essay_cols].fillna("").apply(lambda s: s.str.len()).sum(axis=1)
    out["essays_written"] = (out[essay_cols].fillna("").apply(lambda s: s.str.len()) > 0).sum(axis=1)
    return out


def add_profile_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """0-1 score: fraction of meaningful profile fields filled in.

    Counts a field as 'filled' if non-null AND not equal to 'unknown' sentinel.
    """
    out = df.copy()

    fields = [
        "body_type", "diet", "drinks", "drugs", "education",
        "ethnicity", "height", "income", "job", "offspring",
        "pets", "religion", "sign", "smokes", "speaks",
        "essay0", "essay1", "essay2", "essay3", "essay4",
    ]
    available = [f for f in fields if f in out.columns]

    def is_filled(value, col):
        if pd.isna(value):
            return False
        if col.startswith("essay"):
            return len(str(value).strip()) > 0
        if isinstance(value, str) and value.lower() == "unknown":
            return False
        return True

    completeness = pd.DataFrame({col: out[col].apply(lambda v, c=col: is_filled(v, c)) for col in available})
    out["profile_completeness"] = completeness.mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# Ordinal scales for variables that are nominally categorical but inherently ordered
# ---------------------------------------------------------------------------

DRINKS_ORDER = {
    "not at all": 0,
    "rarely": 1,
    "socially": 2,
    "often": 3,
    "very often": 4,
    "desperately": 5,
}

SMOKES_ORDER = {
    "no": 0,
    "trying to quit": 1,
    "when drinking": 2,
    "sometimes": 3,
    "yes": 4,
}

EDUCATION_RANKING = {
    "graduated from high school": 1,
    "working on high school": 0,
    "high school": 1,
    "graduated from two-year college": 2,
    "working on two-year college": 2,
    "two-year college": 2,
    "graduated from college/university": 3,
    "working on college/university": 3,
    "college/university": 3,
    "graduated from masters program": 4,
    "working on masters program": 4,
    "masters program": 4,
    "graduated from law school": 5,
    "graduated from med school": 5,
    "graduated from ph.d program": 5,
    "working on ph.d program": 5,
    "ph.d program": 5,
}


def add_ordinal_scales(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "drinks" in out.columns:
        out["drinks_score"] = out["drinks"].map(DRINKS_ORDER)
    if "smokes" in out.columns:
        out["smokes_score"] = out["smokes"].map(SMOKES_ORDER)
    if "education" in out.columns:
        out["education_score"] = out["education"].str.lower().map(
            lambda v: EDUCATION_RANKING.get(v, np.nan) if isinstance(v, str) else np.nan
        )
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature-engineering pipeline."""
    return (
        df.pipe(add_age_group)
        .pipe(add_bio_length)
        .pipe(add_profile_completeness)
        .pipe(add_ordinal_scales)
    )

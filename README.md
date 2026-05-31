# OkCupid User Analytics

Exploratory analysis of ~60K OkCupid dating profiles — data cleaning, feature engineering, correlation analysis, and cohort comparisons — packaged as Jupyter notebooks plus an interactive Streamlit dashboard. Dashboard - https://okcupid-analytics.streamlit.app/

## Highlights

- Pearson correlation analysis (with 95% confidence intervals via Fisher z-transform) of bio length vs income, including cohort-conditional breakdowns by sex and age group.
- Profile-completeness scoring across 20 fields and ordinal scales for `drinks`, `smokes`, `education`.
- Welch's t-tests for cohort comparisons (e.g. male vs female bio length).
- Streamlit dashboard with 4 tabs: Overview, Correlations, Cohort Comparison, Bio Analysis.

## Stack

Pandas · NumPy · SciPy · scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit · Jupyter

## Project layout

```
src/
  cleaning.py   # imputation, outlier capping, implausible-row removal, quality report
  features.py   # age groups, bio metrics, completeness score, ordinal scales
  analysis.py   # Pearson correlation w/ CI, cohort_pearson, ttest_two_groups
notebooks/
  01_exploration.ipynb           # EDA: schema, distributions, missingness
  02_cleaning.ipynb              # cleaning pipeline + before/after quality stats
  03_feature_engineering.ipynb   # derived features
  04_correlation_analysis.ipynb  # correlation matrix + cohort-conditional r
  05_cohort_analysis.ipynb       # group comparisons, t-tests, heatmaps
streamlit_app/
  app.py
data/
  raw/         # OkCupid CSV (gitignored — see data/README.md to download)
  processed/   # cleaned parquet
```

## Derived features

| Feature                | Description                                              |
|------------------------|----------------------------------------------------------|
| `age_group`            | Binned: 18–24, 25–29, 30–34, 35–44, 45+                  |
| `bio_length`           | Character count of `essay0`                              |
| `bio_word_count`       | Word count of `essay0`                                   |
| `total_essay_length`   | Sum of all 10 essay fields                               |
| `essays_written`       | Count of non-empty essays (0–10)                         |
| `profile_completeness` | Fraction of 20 meaningful fields filled (0–1)            |
| `drinks_score`         | Ordinal 0–5 (not at all → desperately)                   |
| `smokes_score`         | Ordinal 0–4 (no → yes)                                   |
| `education_score`      | Ordinal 0–5 (high school → PhD / Law / Med)              |

## Cleaning pipeline

1. **Sentinel recoding** — `income == -1` (undisclosed) → `NaN`.
2. **Imputation** — median for numeric, `"unknown"` for categorical, empty string for essays.
3. **Outlier capping** — Winsorize at 0.5% / 99.5% quantiles.
4. **Implausible row removal** — age outside 18–100, height outside 50"–84".

`quality_report` prints rows-fully-complete and rows-with-core-fields-complete before vs after.

## Dataset

[OkCupid Profiles](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles) on Kaggle — ~60K dating profiles with 31 attributes (demographics, lifestyle, 10 essay fields).

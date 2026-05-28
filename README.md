# OkCupid User Analytics

End-to-end exploratory analysis of ~60K OkCupid dating profiles. Demonstrates the data-engineering and statistical analysis stack: **Pandas / NumPy** for cleaning, imputation, outlier handling, and feature engineering; **SciPy** for correlation and hypothesis testing; **Matplotlib / Seaborn / Plotly** for visualization; and an interactive **Streamlit** dashboard.

## Headline analysis

The original motivation was to investigate whether **profile bio length correlates with income** — a hypothesis explored on Bumble's dataset (proprietary) and replicated here using the publicly-available OkCupid Profiles dataset on Kaggle.

The notebooks compute:
- Overall Pearson correlation between income and bio length (with 95% CI)
- Cohort-conditional correlations (by sex, age group)
- Full pairwise correlation matrix across 11 numeric features
- Cohort comparisons via t-tests and grouped aggregates

Findings are reported honestly — see `notebooks/04_correlation_analysis.ipynb` for the actual r value, p-value, and confidence interval on your run.

## Project structure

```
okcupid-user-analytics/
├── data/
│   ├── README.md             # How to download the OkCupid dataset
│   ├── raw/                  # Raw CSV (gitignored)
│   └── processed/            # Cleaned + featured parquet (gitignored, regenerable)
├── notebooks/
│   ├── 01_exploration.ipynb          # Schema, distributions, missingness
│   ├── 02_cleaning.ipynb             # Imputation, outlier capping, quality report
│   ├── 03_feature_engineering.ipynb  # age_group, bio metrics, completeness, ordinal scales
│   ├── 04_correlation_analysis.ipynb # Pearson r — overall, by sex, by age group
│   └── 05_cohort_analysis.ipynb      # Cohort comparisons + cross-cohort heatmaps
├── src/
│   ├── cleaning.py    # impute_missing, cap_outliers, remove_implausible, quality reporting
│   ├── features.py    # age groups, bio length, profile completeness, ordinal scales
│   └── analysis.py    # pearson_correlation (with CI), cohort_pearson, ttest_two_groups
├── streamlit_app/
│   ├── app.py                # Interactive dashboard
│   └── requirements.txt      # Lightweight deps for Streamlit Cloud deployment
├── requirements.txt
└── .gitignore
```

## Feature engineering

Beyond the raw columns, the pipeline derives:

| Feature | Description |
|---|---|
| `age_group` | Binned: 18-24, 25-29, 30-34, 35-44, 45+ |
| `bio_length` | Character count of `essay0` (the "About me" field) |
| `bio_word_count` | Word count of `essay0` |
| `total_essay_length` | Sum of all 10 essay fields |
| `essays_written` | Count of non-empty essays (out of 10) |
| `profile_completeness` | 0-1 fraction of meaningful fields filled in |
| `drinks_score` | Ordinal 0-5 (not at all → desperately) |
| `smokes_score` | Ordinal 0-4 (no → yes) |
| `education_score` | Ordinal 0-5 (HS → PhD/Law/Med) |

## Cleaning strategy

1. **Sentinel recoding**: `income == -1` (not disclosed) → `NaN`
2. **Imputation**:
   - Numeric (`height`): median
   - Categorical: `"unknown"` sentinel
   - Essays: empty string
3. **Outlier capping**: Winsorize at 0.5% / 99.5% quantiles
4. **Implausible removal**: Age outside 18-100, height outside 50"-84"

The pipeline reports % rows fully complete and % rows with core fields complete before vs after.

## Quick start

```bash
git clone https://github.com/Tavish1510/okcupid-user-analytics.git
cd okcupid-user-analytics

# Set up venv and install deps
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Mac/Linux
pip install -r requirements.txt

# Download dataset — see data/README.md
# (manual download from Kaggle, or use kaggle CLI)

# Run the notebooks in order
jupyter notebook notebooks/

# Launch the Streamlit dashboard
cd streamlit_app
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Sign in to https://share.streamlit.io with your GitHub account
3. **New app** → select this repo
4. Main file: `streamlit_app/app.py`
5. Requirements: `streamlit_app/requirements.txt`
6. Once deployed, upload `data/processed/okcupid_features.parquet` via the Streamlit Cloud "Manage app" → "Files" panel (or commit a slim sampled version to the repo)

## Skills demonstrated

- **Pandas / NumPy**: bulk dataframe transformations, groupby, pivot tables, missing-value strategies
- **Statistical analysis**: Pearson correlation with 95% CI (Fisher z-transform), Welch's t-test, cohort-conditional analysis
- **Feature engineering**: binning, ordinal scaling, composite scores (completeness)
- **Visualization**: Matplotlib (notebooks), Plotly (interactive dashboard), Seaborn
- **Project structure**: reusable `src/` modules, clean separation of cleaning / features / analysis, deployable Streamlit app

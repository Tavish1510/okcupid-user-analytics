import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import cohort_pearson, pearson_correlation
from src.features import engineer_features

st.set_page_config(page_title="OkCupid User Analytics", page_icon="💘", layout="wide")

st.title("OkCupid User Analytics Dashboard")
st.caption("Interactive exploration of ~60K dating profiles — demographics, correlations, and cohort patterns")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "okcupid_features.parquet"
RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "okcupid_profiles.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_parquet(DATA_PATH)
    if RAW_PATH.exists():
        from src.cleaning import impute_missing, cap_outliers, remove_implausible
        df = pd.read_csv(RAW_PATH)
        df = remove_implausible(cap_outliers(impute_missing(df)))
        return engineer_features(df)
    return pd.DataFrame()


df = load_data()
if df.empty:
    st.error(
        "No data found. Download `okcupid_profiles.csv` into `data/raw/` "
        "or run the notebooks to generate `data/processed/okcupid_features.parquet`. "
        "See `data/README.md` for instructions."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    sex_options = ["All"] + sorted([s for s in df["sex"].dropna().unique()])
    sex = st.selectbox("Sex", sex_options)

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))

    require_income = st.checkbox("Only profiles with disclosed income", value=False)
    require_bio = st.checkbox("Only profiles with non-empty bio", value=False)

filtered = df[df["age"].between(age_range[0], age_range[1])]
if sex != "All":
    filtered = filtered[filtered["sex"] == sex]
if require_income:
    filtered = filtered[filtered["income"].notna()]
if require_bio:
    filtered = filtered[filtered["bio_length"] > 0]

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Profiles", f"{len(filtered):,}")
c2.metric("Median age", f"{filtered['age'].median():.0f}")
c3.metric("Median bio length", f"{filtered['bio_length'].median():.0f} chars")
c4.metric("Mean profile completeness", f"{filtered['profile_completeness'].mean():.1%}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_corr, tab_cohort, tab_bio = st.tabs(["Overview", "Correlations", "Cohort comparison", "Bio analysis"])

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered, x="age", nbins=50, color="sex", barmode="overlay",
                            opacity=0.6, title="Age distribution by sex",
                            color_discrete_map={"m": "#4A90E2", "f": "#F5A623"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        comp_bins = pd.cut(filtered["profile_completeness"], bins=10).value_counts().sort_index()
        comp_df = pd.DataFrame({"bin": comp_bins.index.astype(str), "users": comp_bins.values})
        fig = px.bar(comp_df, x="bin", y="users", title="Profile completeness distribution",
                      color_discrete_sequence=["#7ED321"])
        fig.update_xaxes(tickangle=20)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Drinking frequency")
        drinks_counts = filtered["drinks"].value_counts()
        fig = px.pie(values=drinks_counts.values, names=drinks_counts.index, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.subheader("Status")
        status_counts = filtered["status"].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)


with tab_corr:
    st.subheader("Pearson correlation analysis")

    numeric_cols = ["age", "height", "income", "bio_length", "bio_word_count", "total_essay_length",
                    "essays_written", "profile_completeness", "drinks_score", "smokes_score", "education_score"]
    available_numeric = [c for c in numeric_cols if c in filtered.columns]

    c1, c2 = st.columns(2)
    with c1:
        x_var = st.selectbox("X variable", available_numeric, index=available_numeric.index("income"))
    with c2:
        y_var = st.selectbox("Y variable", available_numeric, index=available_numeric.index("bio_length"))

    result = pearson_correlation(filtered, x_var, y_var)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pearson r", f"{result['r']:.4f}")
    c2.metric("95% CI", f"[{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
    c3.metric("n", f"{result['n']:,}")

    sub = filtered[[x_var, y_var]].dropna()
    if len(sub) > 0:
        # Sample for plotting if dataset is large
        plot_sub = sub.sample(min(8000, len(sub)), random_state=42)
        fig = px.scatter(plot_sub, x=x_var, y=y_var, opacity=0.3,
                          trendline="lowess", trendline_color_override="red",
                          title=f"{x_var} vs {y_var}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full correlation matrix")
    corr = filtered[available_numeric].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto", zmin=-0.5, zmax=0.5,
                     text_auto=".2f", title="Pearson correlation matrix")
    st.plotly_chart(fig, use_container_width=True)


with tab_cohort:
    st.subheader("Cohort breakdown")

    c1, c2 = st.columns(2)
    with c1:
        group_col = st.selectbox("Group by", ["age_group", "sex", "education_score", "drinks", "smokes"])
    with c2:
        metric = st.selectbox("Metric", ["bio_length", "profile_completeness", "income", "essays_written"])

    if group_col in filtered.columns:
        cohort = filtered.dropna(subset=[metric]).groupby(group_col, observed=True).agg(
            mean=(metric, "mean"),
            median=(metric, "median"),
            count=(metric, "count"),
        ).reset_index()
        cohort[group_col] = cohort[group_col].astype(str)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(cohort, x=group_col, y="median", title=f"Median {metric} by {group_col}",
                          color_discrete_sequence=["#4A90E2"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(cohort, x=group_col, y="count", title=f"Users per cohort",
                          color_discrete_sequence=["#7ED321"])
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(cohort.round(2), use_container_width=True)


with tab_bio:
    st.subheader("Bio (essay0) text analysis")

    bios = filtered[filtered["bio_length"] > 0]
    st.write(f"Bios analyzed: {len(bios):,}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(bios[bios["bio_length"] < 3000], x="bio_length", nbins=50,
                            title="Bio length distribution (clipped at 3000 chars)",
                            color_discrete_sequence=["#9013FE"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(bios[bios["bio_word_count"] < 500], x="bio_word_count", nbins=50,
                            title="Bio word count distribution (clipped at 500)",
                            color_discrete_sequence=["#F5A623"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample bios")
    if len(bios) > 0:
        n_samples = st.slider("Number of samples", 1, 10, 5)
        samples = bios.sample(min(n_samples, len(bios)), random_state=int(np.random.randint(0, 1e6))).reset_index(drop=True)
        for i, row in samples.iterrows():
            with st.expander(f"#{i+1} — age {row['age']}, {row['sex']}, bio length {row['bio_length']} chars"):
                st.write(row["essay0"])

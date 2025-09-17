# streamlit_app.py
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# -----------------------------
# Page config & theme
# -----------------------------
st.set_page_config(
    page_title="MovieLens 200k • Ratings Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
alt.theme.enable("dark")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure types
    if "timestamp" in df.columns:
        # if timestamp is string like '1997-12-04 15:55:49'
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # year / decade may be float in the sample — coerce to Int64
    for col in ["year", "decade", "rating_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # normalize genres (pipe-separated)
    df["genres"] = df["genres"].fillna("")
    df["genre_list"] = df["genres"].apply(lambda s: [g.strip() for g in s.split("|")] if "|" in s else ([s] if s else []))
    # explode for genre-level analyses
    df_exploded = df.explode("genre_list", ignore_index=True)
    df_exploded.rename(columns={"genre_list": "genre"}, inplace=True)
    return df, df_exploded

def fmt_pct(x):
    return f"{x:.1f}%"

def filter_df(df: pd.DataFrame,
              gender_sel, age_range, occ_sel, year_range, rating_year_range, genres_sel, exploded=False):
    d = df.copy()
    # Demographic filters
    if gender_sel:
        d = d[d["gender"].isin(gender_sel)]
    if age_range:
        d = d[(d["age"] >= age_range[0]) & (d["age"] <= age_range[1])]
    if occ_sel:
        d = d[d["occupation"].isin(occ_sel)]
    # Movie release year filter
    if "year" in d.columns and year_range:
        d = d[d["year"].between(year_range[0], year_range[1])]
    # Rating year filter (when rating was given)
    if "rating_year" in d.columns and rating_year_range:
        d = d[d["rating_year"].between(rating_year_range[0], rating_year_range[1])]
    # Genre filter (only meaningful if exploded or if you want “movie has any of selected genres”)
    if genres_sel and len(genres_sel) > 0:
        if exploded:
            d = d[d["genre"].isin(genres_sel)]
        else:
            # keep rows where movie has ANY selected genre
            d = d[d["genre_list"].apply(lambda gs: any(g in gs for g in genres_sel))]
    return d

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = "../data/movie_ratings.csv"
df, df_exploded = load_data(DATA_PATH)

# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar:
    st.title("🎬 MovieLens 200k")
    st.caption("Interactive filters apply to all charts below.")

    # Gender
    genders = sorted([g for g in df["gender"].dropna().unique()])
    gender_sel = st.multiselect("Gender", options=genders, default=[])

    # Age
    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max), step=1)

    # Occupation
    occs = sorted([o for o in df["occupation"].dropna().unique()])
    occ_sel = st.multiselect("Occupation", options=occs, default=[])

    # Movie release year
    year_vals = df["year"].dropna().astype(int)
    yr_min, yr_max = (int(year_vals.min()), int(year_vals.max())) if not year_vals.empty else (1900, 2020)
    year_range = st.slider("Movie release year", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max), step=1)

    # Rating year
    ryr_vals = df["rating_year"].dropna().astype(int)
    ryr_min, ryr_max = (int(ryr_vals.min()), int(ryr_vals.max())) if not ryr_vals.empty else (1995, 2020)
    rating_year_range = st.slider("Rating year (year when rating was given)",
                                  min_value=ryr_min, max_value=ryr_max, value=(ryr_min, ryr_max), step=1)

    # Genres
    all_genres = sorted([g for g in df_exploded["genre"].dropna().unique()])
    genres_sel = st.multiselect("Genres (filter)", options=all_genres, default=[])

    st.markdown("---")
    st.subheader("Sampling thresholds")
    min_n_genre = st.slider("Min ratings per genre (for satisfaction ranks)", 10, 500, 50, 10)
    min_n_movie_1 = st.slider("Min ratings per movie (Top list #1)", 10, 500, 50, 10)
    min_n_movie_2 = st.slider("Min ratings per movie (Top list #2)", 10, 1000, 150, 10)

# Apply filters
df_film = filter_df(df, gender_sel, age_range, occ_sel, year_range, rating_year_range, genres_sel, exploded=False)
df_gen = filter_df(df_exploded, gender_sel, age_range, occ_sel, year_range, rating_year_range, genres_sel, exploded=True)

# -----------------------------
# Header & data peek
# -----------------------------
st.markdown("## 🎥 MovieLens Data Analysis Dashboard")
st.caption("Dataset: MovieLens 200k (sample with user demographics, movie metadata, and ratings).")

with st.expander("Preview dataset (filtered)", expanded=False):
    st.dataframe(df_film.head(25), width=True)

kpi_col = st.columns(4)
with kpi_col[0]:
    st.metric("Ratings (filtered)", value=f"{len(df_film):,}")
with kpi_col[1]:
    st.metric("Unique users", value=f"{df_film['user_id'].nunique():,}")
with kpi_col[2]:
    st.metric("Unique movies", value=f"{df_film['movie_id'].nunique():,}")
with kpi_col[3]:
    st.metric("Mean rating", value=f"{df_film['rating'].mean():.2f}")

st.markdown("---")

# -----------------------------
# Q1: Genre breakdown (rated movies)
# -----------------------------
st.subheader("Q1. What’s the breakdown of genres for the movies that were rated?")
st.caption("Movies can belong to multiple genres. We **explode genres** here to profile preferences. Include counts to avoid misinterpretation.")

if df_gen.empty:
    st.info("No data after filters. Try broadening your selections.")
else:
    genre_counts = (
        df_gen
        .groupby("genre", dropna=True)
        .agg(n_ratings=("rating", "size"),
             n_movies=("movie_id", "nunique"))
        .reset_index()
        .sort_values("n_ratings", ascending=False)
    )
    total = genre_counts["n_ratings"].sum()
    genre_counts["share"] = 100.0 * genre_counts["n_ratings"] / max(total, 1)

    chart = (
        alt.Chart(genre_counts)
        .mark_bar()
        .encode(
            x=alt.X("n_ratings:Q", title="Ratings count"),
            y=alt.Y("genre:N", sort="-x", title="Genre"),
            tooltip=["genre", "n_ratings", "n_movies", alt.Tooltip("share:Q", format=".1f")]
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Interpretation tip: Because movies may have multiple genres, counts can exceed total ratings. Use **n_movies** and **share** for context.")

st.markdown("---")

# -----------------------------
# Q2: Which genres have the highest viewer satisfaction?
# -----------------------------
st.subheader("Q2. Which genres have the highest viewer satisfaction (highest ratings)?")
st.caption(f"We compute mean rating by genre with a minimum of **n ≥ {min_n_genre} ratings** to reduce small-sample noise.")

if df_gen.empty:
    st.info("No data after filters.")
else:
    genre_quality = (
        df_gen.groupby("genre", dropna=True)
        .agg(
            mean_rating=("rating", "mean"),
            n=("rating", "size")
        )
        .reset_index()
    )
    genre_quality = genre_quality[genre_quality["n"] >= min_n_genre].sort_values("mean_rating", ascending=False)

    if genre_quality.empty:
        st.warning("No genres meet the minimum rating threshold. Lower the slider in the sidebar.")
    else:
        chart2 = (
            alt.Chart(genre_quality)
            .mark_bar()
            .encode(
                x=alt.X("mean_rating:Q", scale=alt.Scale(domain=[1, 5]), title="Mean rating (1–5)"),
                y=alt.Y("genre:N", sort="-x", title="Genre"),
                color=alt.condition(alt.datum.mean_rating >= 4.0, alt.value("#3CB371"), alt.value("#1f77b4")),
                tooltip=["genre", alt.Tooltip("mean_rating:Q", format=".2f"), "n"]
            )
            .properties(height=400)
        )
        st.altair_chart(chart2, use_container_width=True)
        st.caption("Note: Results depend on your filters. A few high-rated genres may still reflect user or era biases (check **n**).")

st.markdown("---")

# -----------------------------
# Q3: How does mean rating change across release years?
# -----------------------------
st.subheader("Q3. How does mean rating change across movie release years?")
st.caption("We compute mean rating by **release year**. Hover to see counts; sparse years can be noisy.")

if df_film.empty or df_film["year"].dropna().empty:
    st.info("No data after filters.")
else:
    by_year = (
        df_film.dropna(subset=["year"])
        .groupby("year")
        .agg(mean_rating=("rating", "mean"),
             n=("rating", "size"))
        .reset_index()
        .sort_values("year")
    )
    line = (
        alt.Chart(by_year)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Release year"),
            y=alt.Y("mean_rating:Q", scale=alt.Scale(domain=[1, 5]), title="Mean rating"),
            tooltip=["year", alt.Tooltip("mean_rating:Q", format=".2f"), "n"]
        )
        .properties(height=350)
    )
    st.altair_chart(line, use_container_width=True)
    st.caption("Consider pairing with filters (e.g., rating year) to separate nostalgia effects from contemporary sentiment.")

st.markdown("---")

# -----------------------------
# Q4: Best-rated movies at scale
# -----------------------------
st.subheader("Q4. What are the best-rated movies with sufficient ratings?")
st.caption("We rank movies by mean rating with minimum sample thresholds to avoid small-sample artifacts.")

def best_movies(df_in: pd.DataFrame, min_n: int):
    g = (
        df_in.groupby(["movie_id", "title"], dropna=False)
        .agg(
            mean_rating=("rating", "mean"),
            n=("rating", "size"),
            year=("year", "min")
        )
        .reset_index()
    )
    g = g[g["n"] >= min_n].sort_values(["mean_rating", "n"], ascending=[False, False])
    return g

top_50 = best_movies(df_film, min_n=min_n_movie_1).head(5)
top_150 = best_movies(df_film, min_n=min_n_movie_2).head(5)

cols_top = st.columns(2)
with cols_top[0]:
    st.markdown(f"**Top 5 (n ≥ {min_n_movie_1})**")
    if top_50.empty:
        st.info("No movies meet the threshold.")
    else:
        st.dataframe(
            top_50.assign(mean_rating=lambda d: d["mean_rating"].round(3)),
            width=True,
            hide_index=True,
            column_order=["title", "year", "mean_rating", "n"],
        )

with cols_top[1]:
    st.markdown(f"**Top 5 (n ≥ {min_n_movie_2})**")
    if top_150.empty:
        st.info("No movies meet the threshold.")
    else:
        st.dataframe(
            top_150.assign(mean_rating=lambda d: d["mean_rating"].round(3)),
            width=True,
            hide_index=True,
            column_order=["title", "year", "mean_rating", "n"],
        )

st.markdown("---")

# -----------------------------
# Insights box
# -----------------------------
st.subheader("Insights & Notes")
st.write(
    """
- **Genre breakdown** shows where engagement concentrates; because movies can have multiple genres, genre counts reflect *preference profiling* rather than market share.
- **Viewer satisfaction by genre** uses a minimum ratings threshold to reduce small-sample noise; try increasing the threshold if rankings look unstable.
- **Mean rating over release years** can reflect both film quality and user/rater composition; consider filtering by **rating_year** to isolate period effects.
- **Best-rated movies** are sensitive to thresholds; a movie with 20 perfect ratings may be less reliable than one with 500 good ratings.
"""
)

# Optional: CSV downloads
dl_col = st.columns(3)
with dl_col[0]:
    st.download_button("Download filtered (movie-level) CSV", df_film.to_csv(index=False).encode(), "filtered_movies.csv")
with dl_col[1]:
    if not df_gen.empty:
        st.download_button("Download filtered (genre-exploded) CSV", df_gen.to_csv(index=False).encode(), "filtered_genre_rows.csv")
with dl_col[2]:
    pass

st.caption("Built with Streamlit, pandas, and Altair. Data: MovieLens 200k (educational use).")

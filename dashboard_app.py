import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Udemy Finance & Accounting Dashboard", layout="wide")

st.title("Udemy Finance & Accounting Courses Dashboard")
st.write("Simple version without Plotly for easy running")

# ---- Load Data ----
@st.cache_data
def load_data():
    # Replace with your CSV file name
    df = pd.read_csv("udemy_data.csv")
    df = df.dropna()
    if "published_time" in df.columns:
        df["published_year"] = pd.to_datetime(df["published_time"], errors="coerce").dt.year
    else:
        df["published_year"] = 0
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---- Filters ----
st.sidebar.header("Filters")
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.0, 0.1)
min_lectures = st.sidebar.slider(
    "Minimum Lectures",
    0,
    int(df["num_published_lectures"].max()),
    0,
    1,
)
filtered_df = df[(df["avg_rating"] >= min_rating) &
                 (df["num_published_lectures"] >= min_lectures)]

st.write(f"Rows after filtering: {filtered_df.shape[0]}")
st.dataframe(filtered_df.head())

# ---- Quick Stats ----
if not filtered_df.empty:
    st.subheader("Summary Statistics")
    st.dataframe(
        filtered_df[["num_subscribers",
                     "avg_rating",
                     "num_reviews",
                     "num_published_lectures"]].describe()
    )

    st.subheader("Average Rating Histogram")
    st.bar_chart(np.histogram(filtered_df["avg_rating"], bins=20)[0])
else:
    st.info("No data after filtering.")

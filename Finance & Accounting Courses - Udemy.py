import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import io

# Streamlit page configuration
st.set_page_config(page_title="Udemy Finance & Accounting Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 36px; color: #2c3e50; text-align: center; font-weight: bold;}
    .sub-header {font-size: 24px; color: #34495e; font-weight: bold;}
    .metric-box {background-color: #ecf0f1; padding: 10px; border-radius: 5px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# Step 1: Load and Preprocess Data (before sidebar)
@st.cache_data
def load_data():
    df = pd.read_csv('udemy_data.csv')  # <-- Make sure this file name matches your actual file
    df = df.dropna()
    if 'is_wishlisted' in df.columns:
        df = pd.get_dummies(df, columns=['is_wishlisted'], drop_first=True)
    if 'published_time' in df.columns:
        try:
            df['published_year'] = pd.to_datetime(df['published_time'], errors='coerce').dt.year
        except Exception as e:
            st.warning(f"Could not parse 'published_time' column: {e}")
            df['published_year'] = 0
    else:
        st.warning("'published_time' column not found. Trend analysis will be skipped.")
        df['published_year'] = 0
    return df

# Safely load the data
try:
    df = load_data()
    st.write("✅ Data loaded successfully!", df.shape)
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Sidebar for filters
st.sidebar.header("Filters & Settings")
min_rating = st.sidebar.slider("Minimum Average Rating", 0.0, 5.0, 3.0, 0.1)
max_lectures = int(df['num_published_lectures'].max()) if not df['num_published_lectures'].empty else 0
min_lectures = st.sidebar.slider("Minimum Number of Lectures", 0, max_lectures, 0, 1)
sort_by = st.sidebar.selectbox("Sort Top Courses By", ["num_subscribers", "avg_rating", "num_reviews"])

# Title
st.markdown('<p class="main-header">Udemy Finance & Accounting Courses Dashboard</p>', unsafe_allow_html=True)
st.markdown("Built for Internship Presentation - June 2025")

# Apply filters
filtered_df = df[(df['avg_rating'] >= min_rating) & (df['num_published_lectures'] >= min_lectures)]
st.write(f"🔍 Showing {filtered_df.shape[0]} rows after filtering")

# Step 2: Train Model
@st.cache_data
def train_model(df):
    required_features = ['avg_rating', 'num_reviews', 'num_published_lectures']
    missing_features = [feat for feat in required_features if feat not in df.columns]
    if missing_features:
        st.error(f"Missing required features: {missing_features}")
        return None, None, None
    X = df[required_features]
    y = df['num_subscribers']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    return mse, r2, predictions

# Run model if data exists
if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust the filters and try again.")
    mse, r2, predictions = None, None, None
else:
    mse, r2, predictions = train_model(filtered_df)

# Model Performance Section
st.markdown('<p class="sub-header">Model Performance</p>', unsafe_allow_html=True)
if mse is not None and r2 is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown(f"Mean Squared Error<br>{mse:,.2f}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown(f"R-squared<br>{r2:.2f}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Model performance metrics unavailable due to empty filtered data.")

# Dataset Insights Section
st.markdown('<p class="sub-header">Dataset Insights</p>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    st.write("Summary Statistics")
    if not filtered_df.empty:
        summary = filtered_df[['num_subscribers', 'avg_rating', 'num_reviews', 'num_published_lectures']].describe()
        st.dataframe(summary.style.format("{:.2f}"))
    else:
        st.info("No data to display summary statistics.")
with col4:
    st.write("Average Rating Distribution")
    if not filtered_df.empty:
        fig_hist = px.histogram(filtered_df, x='avg_rating', nbins=20, title="Distribution of Average Ratings")
        fig_hist.update_layout(xaxis_title="Average Rating", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No data to display histogram.")

# Model Predictions Section
st.markdown('<p class="sub-header">Model Predictions</p>', unsafe_allow_html=True)
if predictions is not None:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=predictions['Actual'], y=predictions['Predicted'], mode='markers', 
                                     marker=dict(size=8, opacity=0.5), name='Predictions'))
    fig_scatter.add_trace(go.Scatter(x=[predictions['Actual'].min(), predictions['Actual'].max()],
                                     y=[predictions['Actual'].min(), predictions['Actual'].max()],
                                     mode='lines', line=dict(color='red', dash='dash'), name='Ideal'))
    fig_scatter.update_layout(title="Actual vs Predicted Subscribers", xaxis_title="Actual Subscribers", 
                              yaxis_title="Predicted Subscribers")
    st.plotly_chart(fig_scatter, use_container_width=True)

    buffer = io.StringIO()
    predictions.to_csv(buffer, index=False)
    st.download_button(label="Download Predictions", data=buffer.getvalue(), file_name="predictions.csv", mime="text/csv")
else:
    st.info("Model predictions unavailable due to empty filtered data.")

# Top Courses Section
st.markdown('<p class="sub-header">Top Courses Analysis</p>', unsafe_allow_html=True)
top_n = st.slider("Number of Top Courses to Display", 5, 20, 10)
if not filtered_df.empty:
    top_courses = filtered_df.nlargest(top_n, sort_by).sort_values(sort_by, ascending=True)
    fig_bar = px.bar(top_courses, x=sort_by, y='title', orientation='h',
                     title=f"Top {top_n} Courses by {sort_by.replace('_', ' ').title()}",
                     labels={'title': 'Course Title', sort_by: sort_by.replace('_', ' ').title()},
                     color=sort_by)
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No data to display top courses.")

# Trend Analysis Section
st.markdown('<p class="sub-header">Trends Over Time</p>', unsafe_allow_html=True)
if 'published_year' in filtered_df.columns and filtered_df['published_year'].notna().any():
    trend_data = filtered_df.groupby('published_year').agg({'num_subscribers': 'mean', 'avg_rating': 'mean'}).reset_index()
    if not trend_data.empty:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend_data['published_year'], y=trend_data['num_subscribers'], mode='lines+markers', 
                                       name='Avg Subscribers'))
        fig_trend.add_trace(go.Scatter(x=trend_data['published_year'], y=trend_data['avg_rating'] * 1000, mode='lines+markers', 
                                       name='Avg Rating (scaled)', yaxis='y2'))
        fig_trend.update_layout(
            title="Average Subscribers and Rating by Publication Year",
            xaxis_title="Year",
            yaxis_title="Average Subscribers",
            yaxis2=dict(title="Average Rating (scaled)", overlaying='y', side='right', range=[0, 5000])
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No trend data available after filtering.")
else:
    st.info("Trend analysis unavailable due to missing or invalid 'published_year' data.")

import sys
from pathlib import Path

# Add project root to sys.path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, fetch_predictions

# App setup
st.set_page_config(layout="wide")
st.title("🚴 Citi Bike Prediction Monitor: Mean Absolute Error (MAE) Dashboard")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
past_hours = st.sidebar.slider(
    "⏳ Number of Past Hours to Plot",
    min_value=12,
    max_value=24 * 28,
    value=24,
    step=1,
)

# Fetch data
st.write(f"📥 Fetching Citi Bike predictions and actuals for the past {past_hours} hours...")
df_actual = fetch_hourly_rides(past_hours)
df_pred = fetch_predictions(past_hours)

# Merge data and calculate error
merged_df = pd.merge(df_actual, df_pred, on=["pickup_location_id", "pickup_hour"])
merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

# ----------------------------------------
# MAE Over Time (Global)
# ----------------------------------------
st.subheader("📈 Overall MAE Over Time")
mae_by_hour = (
    merged_df.groupby("pickup_hour")["absolute_error"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_error": "MAE"})
)

fig_overall_mae = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"Mean Absolute Error (MAE) Over Last {past_hours} Hours",
    labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
    markers=True,
    template="plotly_dark",
)
st.plotly_chart(fig_overall_mae, use_container_width=True)

# ----------------------------------------
# MAE by Location (Bar Chart)
# ----------------------------------------
st.subheader("Average MAE by Location")
mae_by_location = (
    merged_df.groupby("pickup_location_id")["absolute_error"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_error": "MAE"})
    .sort_values("MAE", ascending=False)
)

fig_mae_location = px.bar(
    mae_by_location,
    x="pickup_location_id",
    y="MAE",
    title="Mean Absolute Error by Location",
    labels={"pickup_location_id": "Location ID", "MAE": "Mean Absolute Error"},
    template="plotly_dark",
)
st.plotly_chart(fig_mae_location, use_container_width=True)

# ----------------------------------------
# Top 3 High-Demand Locations by MAE Trend
# ----------------------------------------
rides_by_location = (
    merged_df.groupby("pickup_location_id")["rides"]
    .sum()
    .sort_values(ascending=False)
)

top3_ids = rides_by_location.head(3).index.tolist()
top3_df = merged_df[merged_df["pickup_location_id"].isin(top3_ids)]

st.subheader(" MAE Trend for Top 3 High-Demand Locations")
fig_top3 = px.line(
    top3_df,
    x="pickup_hour",
    y="absolute_error",
    color="pickup_location_id",
    title="Top 3 Busy Stations: MAE Over Time",
    labels={"pickup_hour": "Pickup Hour", "absolute_error": "MAE"},
    markers=True,
    template="plotly_dark",
)
st.plotly_chart(fig_top3, use_container_width=True)

# ----------------------------------------
# Location Filter Dropdown
# ----------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Custom Location Filter**")
all_location_ids = sorted(merged_df["pickup_location_id"].unique())
selected_ids = st.sidebar.multiselect("Select Location IDs", all_location_ids, default=top3_ids)

if selected_ids:
    selected_df = merged_df[merged_df["pickup_location_id"].isin(selected_ids)]
    st.subheader("🔍 MAE Trend for Selected Locations")
    fig_filtered = px.line(
        selected_df,
        x="pickup_hour",
        y="absolute_error",
        color="pickup_location_id",
        title="Filtered Location(s): MAE Trend",
        labels={"pickup_hour": "Pickup Hour", "absolute_error": "MAE"},
        markers=True,
        template="plotly_dark",
    )
    st.plotly_chart(fig_filtered, use_container_width=True)
else:
    st.info("Please select at least one location to view MAE trends.")

# ----------------------------------------
# Summary Statistics
# ----------------------------------------
st.subheader("📊 MAE Summary Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Average MAE", f"{merged_df['absolute_error'].mean():.3f}")
col2.metric("Max MAE", f"{merged_df['absolute_error'].max():.3f}")
col3.metric("MAE Std Dev", f"{merged_df['absolute_error'].std():.3f}")

st.markdown(
    "📌 **Note:** A lower MAE indicates higher prediction accuracy. Use this dashboard to identify stations "
    "with high prediction error and validate model improvements."
)

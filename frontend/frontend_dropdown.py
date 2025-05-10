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
st.title("Citi Bike Prediction Monitor: MAE Dashboard")

# Sidebar - Settings
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,
    max_value=24 * 28,
    value=24,
    step=1,
)

top_n_locations = st.sidebar.selectbox("Show Top N Locations", options=[3, 5, 10, 20], index=1)

# Fetch data
st.write(f"Fetching Citi Bike predictions and actuals for the past {past_hours} hours...")
df_actual = fetch_hourly_rides(past_hours)
df_pred = fetch_predictions(past_hours)

# Merge and compute error
merged_df = pd.merge(df_actual, df_pred, on=["pickup_location_id", "pickup_hour"])
merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

# Determine top N default locations by actual rides
top_demand_locations = (
    merged_df.groupby("pickup_location_id")["rides"]
    .sum()
    .sort_values(ascending=False)
    .head(top_n_locations)
    .index.tolist()
)

# Sidebar - Location Filter
all_locations = sorted(merged_df["pickup_location_id"].unique())
selected_locations = st.sidebar.multiselect(
    "Filter by Location ID", all_locations, default=top_demand_locations
)
filtered_df = merged_df[merged_df["pickup_location_id"].isin(selected_locations)]

# MAE by hour
mae_by_hour = (
    filtered_df.groupby("pickup_hour")["absolute_error"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_error": "MAE"})
)

fig_hourly_mae = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"Mean Absolute Error (MAE) for the Past {past_hours} Hours",
    labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
    markers=True,
    template="plotly_dark",
)
st.plotly_chart(fig_hourly_mae, use_container_width=True)

# MAE by location
mae_by_location = (
    filtered_df.groupby("pickup_location_id")["absolute_error"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_error": "MAE"})
)

fig_location_mae = px.bar(
    mae_by_location,
    x="pickup_location_id",
    y="MAE",
    title=" Mean Absolute Error (MAE) by Pickup Location",
    labels={"pickup_location_id": "Location ID", "MAE": "Mean Absolute Error"},
    template="plotly_dark",
)
st.plotly_chart(fig_location_mae, use_container_width=True)

# MAE Summary
st.subheader("MAE Summary Statistics")
col1, col2 = st.columns(2)
col1.metric("Average MAE", f"{mae_by_hour['MAE'].mean():.3f}")
col2.metric("Max MAE", f"{mae_by_hour['MAE'].max():.3f}")

# --------------------------
# Top Locations by Actual and Predicted Rides
# --------------------------
st.subheader("Top Locations by Actual vs Predicted Ride Volume")

# Top locations by actual rides
rides_by_location = (
    merged_df.groupby("pickup_location_id")["rides"]
    .sum()
    .reset_index()
    .rename(columns={"rides": "Total Actual Rides"})
    .sort_values("Total Actual Rides", ascending=False)
)

# Top locations by predicted rides
pred_by_location = (
    merged_df.groupby("pickup_location_id")["predicted_demand"]
    .sum()
    .reset_index()
    .rename(columns={"predicted_demand": "Total Predicted Rides"})
    .sort_values("Total Predicted Rides", ascending=False)
)

# Display side-by-side
col3, col4 = st.columns(2)
col3.subheader("Top by Actual Rides")
col3.dataframe(rides_by_location.head(top_n_locations).reset_index(drop=True), use_container_width=True)

col4.subheader("Top by Predicted Rides")
col4.dataframe(pred_by_location.head(top_n_locations).reset_index(drop=True), use_container_width=True)

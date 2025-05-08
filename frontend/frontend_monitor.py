import sys
from pathlib import Path

# Add project root to sys.path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, fetch_predictions

st.set_page_config(layout="wide")
st.title("Citi Bike Prediction Monitor: Mean Absolute Error (MAE) by Pickup Hour")

# Sidebar for user input
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,
    max_value=24 * 28,  # 4 weeks
    value=24,  # Default
    step=1,
)

# Fetch data
st.write("Fetching Citi Bike predictions and actuals for the past", past_hours, "hours...")
df_actual = fetch_hourly_rides(past_hours)
df_pred = fetch_predictions(past_hours)

# Merge actual and predicted
merged_df = pd.merge(df_actual, df_pred, on=["pickup_location_id", "pickup_hour"])

# Calculate MAE per hour
merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])
mae_by_hour = (
    merged_df.groupby("pickup_hour")["absolute_error"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_error": "MAE"})
)

# Plot
fig = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"Mean Absolute Error (MAE) for the Past {past_hours} Hours",
    labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
    markers=True,
    template="plotly_dark",
)

st.plotly_chart(fig, use_container_width=True)

# Display MAE stats
st.subheader(" MAE Summary")
st.metric("Average MAE", f"{mae_by_hour['MAE'].mean():.3f}")

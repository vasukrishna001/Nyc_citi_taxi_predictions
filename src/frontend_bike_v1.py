import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import zipfile
from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from streamlit_folium import st_folium

from config import DATA_DIR
from inference import (
    get_model_predictions,
    load_batch_of_features_from_store,
    load_metrics_from_registry,
    load_model_from_registry,
)
from plot_utils import plot_aggregated_time_series

# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False


def visualize_predicted_demand(shapefile_path, predicted_demand):
    """
    Visualizes the predicted number of Citi Bike rides on a map of station zones.

    Parameters:
        shapefile_path (str): Path to the Citi Bike zones shapefile.
        predicted_demand (dict): A dictionary where keys are zone IDs (or station IDs)
                                and values are the predicted number of rides.
    """
    gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")

    if "station_id" not in gdf.columns:
        raise ValueError("Shapefile must contain a 'station_id' column to match bike stations.")

    gdf["predicted_demand"] = gdf["station_id"].map(predicted_demand).fillna(0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(
        column="predicted_demand",
        cmap="OrRd",
        linewidth=0.8,
        ax=ax,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
    )
    ax.set_title("Predicted Citi Bike Rides by Station Zone", fontsize=16)
    ax.set_axis_off()
    st.pyplot(fig)


def create_citibike_map(shapefile_path, prediction_data):
    """
    Create an interactive choropleth map of Citi Bike station zones with predicted rides
    """
    zones = gpd.read_file(shapefile_path)

    zones = zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="station_id",
        right_on="pickup_location_id",
        how="left",
    )

    zones["predicted_demand"] = zones["predicted_demand"].fillna(0)
    zones = zones.to_crs(epsg=4326)

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=12, tiles="cartodbpositron")

    colormap = LinearColormap(
        colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
        vmin=zones["predicted_demand"].min(),
        vmax=zones["predicted_demand"].max(),
    )
    colormap.add_to(m)

    def style_function(feature):
        value = feature["properties"].get("predicted_demand", 0)
        return {
            "fillColor": colormap(float(value)),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        zones.to_json(),
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["station_name", "predicted_demand"],
            aliases=["Station:", "Predicted Demand:"],
            style="background-color: white; color: #333333; font-size: 12px; padding: 10px;",
        ),
    ).add_to(m)

    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m


def load_shape_data_file(
    data_dir,
    url="https://example.com/citibike_stations.zip",  # Replace with real Citi Bike shapefile URL
    log=True,
):
    """
    Downloads, extracts, and loads a Citi Bike station zone shapefile as a GeoDataFrame.

    Parameters:
        data_dir (str or Path): Directory where the data will be stored.
        url (str): URL of the shapefile zip file.
        log (bool): Whether to log progress messages.

    Returns:
        GeoDataFrame: The loaded shapefile as a GeoDataFrame.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define paths
    zip_path = data_dir / "citibike_zones.zip"
    extract_path = data_dir / "citibike_zones"
    shapefile_path = extract_path / "citibike_zones.shp"

    # Download the zip file if not already present
    if not zip_path.exists():
        if log:
            print(f"Downloading Citi Bike shapefile from {url}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(response.content)
            if log:
                print(f"Downloaded to {zip_path}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {url}: {e}")
    else:
        if log:
            print(f"Zip file already exists at {zip_path}, skipping download.")

    # Extract if shapefile not present
    if not shapefile_path.exists():
        if log:
            print(f"Extracting Citi Bike zones to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            if log:
                print(f"Extraction complete at {extract_path}")
        except zipfile.BadZipFile as e:
            raise Exception(f"Failed to extract zip file {zip_path}: {e}")
    else:
        if log:
            print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

    # Load and return the shapefile
    if log:
        print(f"Loading Citi Bike zones shapefile from {shapefile_path}...")
    try:
        gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
        if log:
            print("Shapefile successfully loaded.")
        return gdf
    except Exception as e:
        raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")



# Set the Streamlit page configuration
# st.set_page_config(layout="wide")

# Get the current UTC datetime
current_date = pd.Timestamp.now(tz="Etc/UTC")
st.title(f"New York Citi Bike Trip Demand (Next Hour Forecast)")
st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S")}')

# Sidebar progress bar setup
progress_bar = st.sidebar.header("Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 5

# Step 1: Load Citi Bike station shapefile
with st.spinner(text="Downloading Citi Bike station shapefile..."):
    geo_df = load_shape_data_file(DATA_DIR)
    st.sidebar.write("Shapefile successfully loaded.")
    progress_bar.progress(1 / N_STEPS)

# Step 2: Load latest feature batch for inference
with st.spinner(text="Fetching latest batch of features..."):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write("Feature batch loaded.")
    progress_bar.progress(2 / N_STEPS)

# Step 3: Load trained model from registry
with st.spinner(text="Loading prediction model..."):
    model = load_model_from_registry()
    st.sidebar.write("Model loaded from registry.")
    progress_bar.progress(3 / N_STEPS)

# Step 4: Make predictions using the model
with st.spinner(text="Generating predictions..."):
    predictions = get_model_predictions(model, features)
    st.sidebar.write("Predictions generated.")
    progress_bar.progress(4 / N_STEPS)

# Step 5: Visualize results on map
shapefile_path = DATA_DIR / "citibike_zones" / "citibike_zones.shp"  # adjust as needed

with st.spinner(text="Rendering predicted demand on map..."):
    st.subheader("Citi Bike Demand Forecast Map")
    map_obj = create_taxi_map(shapefile_path, predictions)

    if st.session_state.map_created:
        st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

    st.subheader("Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Trips", f"{predictions['predicted_demand'].mean():.0f}")
    with col2:
        st.metric("Maximum Trips", f"{predictions['predicted_demand'].max():.0f}")
    with col3:
        st.metric("Minimum Trips", f"{predictions['predicted_demand'].min():.0f}")

    st.sidebar.write("Map and stats rendering complete.")
    progress_bar.progress(5 / N_STEPS)

# Display top 10 highest demand stations
st.subheader("Top 10 High Demand Stations")
st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))

# Plot historical vs predicted demand for top 10 locations
top10 = (
    predictions.sort_values("predicted_demand", ascending=False).head(10).index.tolist()
)
for location_id in top10:
    fig = plot_aggregated_time_series(
        features=features,
        targets=predictions["predicted_demand"],
        row_id=location_id,
        predictions=pd.Series(predictions["predicted_demand"]),
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

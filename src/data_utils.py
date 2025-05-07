import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import calendar
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytz
import requests

from src.config import RAW_DATA_DIR


def fetch_raw_trip_data(year: int, month: int) -> Path:
    URL = f"https://s3.amazonaws.com/tripdata/JC-{year}{month:02}-citibike-tripdata.csv.zip"
    response = requests.get(URL)

    if response.status_code == 200:
        zip_path = RAW_DATA_DIR / f"JC-{year}{month:02}-citibike-tripdata.csv.zip"
        csv_path = RAW_DATA_DIR / f"JC-{year}{month:02}-citibike-tripdata.csv"
        parquet_path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"

        # Save the ZIP file
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Extract the ZIP file
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)

        # Load CSV and convert to Parquet
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, index=False)

        # Optionally remove the original CSV
        os.remove(csv_path)

        return parquet_path
    else:
        raise Exception(f"{URL} is not available")



def filter_citi_bike_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Filters Citi Bike trip data: drops invalid durations and non-numeric station IDs,
    restricts data to a specific month.

    Args:
        rides (pd.DataFrame): Raw Citi Bike DataFrame.
        year (int): Target year.
        month (int): Target month.

    Returns:
        pd.DataFrame: Filtered rides with [pickup_datetime, pickup_location_id].
    """
    # Time filtering
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = pd.Timestamp(year + (month // 12), (month % 12) + 1, 1)

    # Duration column
    rides["duration"] = pd.to_datetime(rides["ended_at"]) - pd.to_datetime(rides["started_at"])
    rides["duration_minutes"] = rides["duration"].dt.total_seconds() / 60

    # Filters
    duration_filter = (rides["duration"] > pd.Timedelta(0)) & (rides["duration"] <= pd.Timedelta(hours=5))
    date_filter = (pd.to_datetime(rides["started_at"]) >= start_date) & (pd.to_datetime(rides["started_at"]) < end_date)
    location_filter = rides["start_station_id"].notna()

    final_filter = duration_filter & date_filter & location_filter

    total = len(rides)
    valid = final_filter.sum()
    dropped = total - valid
    pct = (dropped / total) * 100

    print(f"Total records: {total:,}")
    print(f"Valid records: {valid:,}")
    print(f"Records dropped: {dropped:,} ({pct:.2f}%)")

    # Final formatting
    filtered = rides[final_filter].copy()
    filtered["pickup_datetime"] = pd.to_datetime(filtered["started_at"])
    filtered["pickup_location_id"] = filtered["start_station_id"].astype(str)

    return filtered[["pickup_datetime", "pickup_location_id"]]




def load_and_process_citibike_data(
    year: int, months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load and process Citi Bike ride data for a specified year and list of months.

    Args:
        year (int): Year to load data for.
        months (Optional[List[int]]): List of months to load. If None, loads all months (1-12).

    Returns:
        pd.DataFrame: Combined and processed Citi Bike ride data for the specified year and months.

    Raises:
        Exception: If no data could be loaded for the specified year and months.
    """
    if months is None:
        months = list(range(1, 13))

    monthly_rides = []

    for month in months:
        file_path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"

        try:
            if not file_path.exists():
                print(f"Downloading Citi Bike data for {year}-{month:02}...")
                fetch_raw_trip_data(year, month)
                print(f"Successfully downloaded data for {year}-{month:02}.")
            else:
                print(f"File already exists for {year}-{month:02}.")

            print(f"Loading Citi Bike data for {year}-{month:02}...")
            rides = pd.read_parquet(file_path, engine="pyarrow")

            rides = filter_citi_bike_data(rides, year, month)
            print(f"Successfully processed data for {year}-{month:02}.")

            monthly_rides.append(rides)

        except FileNotFoundError:
            print(f"File not found for {year}-{month:02}. Skipping...")
        except Exception as e:
            print(f"Error processing data for {year}-{month:02}: {str(e)}")
            continue

    if not monthly_rides:
        raise Exception(
            f"No Citi Bike data could be loaded for the year {year} and specified months: {months}"
        )

    print("Combining all monthly Citi Bike data...")
    combined_rides = pd.concat(monthly_rides, ignore_index=True)
    print("Citi Bike data loading and processing complete!")

    return combined_rides



def fill_missing_rides_full_range(df, hour_col, location_col, rides_col):
    """
    Ensures complete Citi Bike ride data by filling in missing hours and locations with 0 rides.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns [hour_col, location_col, rides_col]
    - hour_col (str): Column containing hourly pickup timestamps (e.g., 'pickup_hour')
    - location_col (str): Column containing station IDs (e.g., 'pickup_location_id')
    - rides_col (str): Column containing ride counts (e.g., 'rides')

    Returns:
    - pd.DataFrame: Completed time series data with all (hour, location) combinations filled.
    """
    # Ensure hour column is in datetime format
    df[hour_col] = pd.to_datetime(df[hour_col])

    # Generate full range of hours from start to end
    full_hours = pd.date_range(
        start=df[hour_col].min(), end=df[hour_col].max(), freq="h"
    )

    # Get all unique station/location IDs
    all_locations = df[location_col].unique()

    # Create full (hour, location) combinations
    full_combinations = pd.DataFrame(
        [(hour, location) for hour in full_hours for location in all_locations],
        columns=[hour_col, location_col]
    )

    # Merge original data with full combinations
    merged_df = pd.merge(full_combinations, df, on=[hour_col, location_col], how="left")

    # Fill missing ride counts with 0
    merged_df[rides_col] = merged_df[rides_col].fillna(0).astype(int)

    return merged_df



def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw Citi Bike ride data into time series format aggregated hourly per location.

    Args:
        rides (pd.DataFrame): DataFrame with 'pickup_datetime' and 'pickup_location_id' columns.

    Returns:
        pd.DataFrame: Hourly time series data with missing hours and locations filled with 0 rides.
    """
    # Round pickup datetime to the nearest hour
    rides["pickup_hour"] = rides["pickup_datetime"].dt.floor("h")

    # Group by hour and location to count rides
    agg_rides = (
        rides.groupby(["pickup_hour", "pickup_location_id"])
        .size()
        .reset_index(name="rides")
    )

    # Fill in missing (hour, location) combinations with 0 rides
    agg_rides_all_slots = (
        fill_missing_rides_full_range(
            agg_rides, "pickup_hour", "pickup_location_id", "rides"
        )
        .sort_values(["pickup_location_id", "pickup_hour"])
        .reset_index(drop=True)
    )

    # Return as-is since pickup_location_id is string (e.g., 'JC072')
    return agg_rides_all_slots



def transform_ts_data_into_features_and_target_loop(
    df, feature_col="rides", window_size=12, step_size=1
):
    """
    Transforms Citi Bike time series data into tabular format for ML training.
    Converts hourly ride counts into lagged feature vectors and targets,
    grouped by each station (pickup_location_id).

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'pickup_hour', 'pickup_location_id', and ride counts.
        feature_col (str): Column to use for time series (default = "rides").
        window_size (int): Number of past hours to use as features (default = 12).
        step_size (int): Number of hours to slide the window forward (default = 1).

    Returns:
        tuple: (features_df, targets_series)
            - features_df includes lagged features, pickup_hour, pickup_location_id
            - targets_series includes the target ride count for prediction
    """
    location_ids = df["pickup_location_id"].unique()
    transformed_data = []

    for location_id in location_ids:
        try:
            location_data = df[df["pickup_location_id"] == location_id].reset_index(drop=True)
            values = location_data[feature_col].values
            times = location_data["pickup_hour"].values

            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            rows = []
            for i in range(0, len(values) - window_size, step_size):
                features = values[i:i + window_size]
                target = values[i + window_size]
                target_time = times[i + window_size]
                row = np.append(features, [target, location_id, target_time])
                rows.append(row)

            feature_columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
            all_columns = feature_columns + ["target", "pickup_location_id", "pickup_hour"]
            transformed_df = pd.DataFrame(rows, columns=all_columns)

            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping location_id {location_id}: {str(e)}")

    if not transformed_data:
        raise ValueError("No data could be transformed. Check data or window size.")

    final_df = pd.concat(transformed_data, ignore_index=True)
    features = final_df[feature_columns + ["pickup_hour", "pickup_location_id"]]
    targets = final_df["target"]

    return features, targets



def transform_ts_data_into_features_and_target(
    df, feature_col="rides", window_size=12, step_size=1
):
    """
    Transforms Citi Bike time series data into a tabular format with lag features and targets.
    Each location's hourly ride data is converted to feature vectors with targets using a sliding window.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'pickup_hour', 'pickup_location_id', and the target feature.
        feature_col (str): Column to be used for features and target (default: "rides").
        window_size (int): Number of past hours to use as features (default: 12).
        step_size (int): Sliding window step (default: 1).

    Returns:
        tuple: (features_df, targets_series, full_transformed_df)
            - features_df: DataFrame with lag features + pickup_hour + pickup_location_id
            - targets_series: Series of target ride values
            - full_transformed_df: Complete DataFrame (features + target + metadata)
    """
    location_ids = df["pickup_location_id"].unique()
    transformed_data = []

    for location_id in location_ids:
        try:
            location_data = df[df["pickup_location_id"] == location_id].reset_index(drop=True)
            values = location_data[feature_col].values
            times = location_data["pickup_hour"].values

            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            rows = []
            for i in range(0, len(values) - window_size, step_size):
                features = values[i:i + window_size]
                target = values[i + window_size]
                target_time = times[i + window_size]
                row = np.append(features, [target, location_id, target_time])
                rows.append(row)

            feature_columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
            all_columns = feature_columns + ["target", "pickup_location_id", "pickup_hour"]
            transformed_df = pd.DataFrame(rows, columns=all_columns)
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping location_id {location_id}: {str(e)}")

    if not transformed_data:
        raise ValueError("No data could be transformed. Check input or window size.")

    final_df = pd.concat(transformed_data, ignore_index=True)
    features = final_df[feature_columns + ["pickup_hour", "pickup_location_id"]]
    targets = final_df["target"]

    return features, targets, final_df



def split_time_series_data(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits Citi Bike time series data into training and testing sets based on a cutoff date.

    Args:
        df (pd.DataFrame): DataFrame with time series features and target.
        cutoff_date (datetime): The date to split into train/test sets.
        target_column (str): Name of the target column (e.g., "target").

    Returns:
        Tuple of train/test features and targets.
    """
    train_data = df[df["pickup_hour"] < cutoff_date].reset_index(drop=True)
    test_data = df[df["pickup_hour"] >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return X_train, y_train, X_test, y_test



def fetch_batch_citibike_data(
    from_date: Union[datetime, str], to_date: Union[datetime, str]
) -> pd.DataFrame:
    """
    Simulate Citi Bike production data by using historical data from exactly 52 weeks ago.

    Args:
        from_date (datetime or str): Start date for the data batch.
        to_date (datetime or str): End date for the data batch.

    Returns:
        pd.DataFrame: Simulated recent ride data shifted from historical period.
    """
    if isinstance(from_date, str):
        from_date = datetime.fromisoformat(from_date)
    if isinstance(to_date, str):
        to_date = datetime.fromisoformat(to_date)

    if from_date >= to_date:
        raise ValueError("'from_date' must be earlier than 'to_date'.")

    # Shift 52 weeks back to sample historical data
    hist_start = from_date - timedelta(weeks=52)
    hist_end = to_date - timedelta(weeks=52)

    rides_start = load_and_process_citibike_data(
        year=hist_start.year, months=[hist_start.month]
    )
    rides_start = rides_start[rides_start["pickup_datetime"] >= hist_start.to_numpy()]

    if hist_end.month != hist_start.month:
        rides_end = load_and_process_citibike_data(
            year=hist_end.year, months=[hist_end.month]
        )
        rides_end = rides_end[rides_end["pickup_datetime"] < hist_end.to_numpy()]
        rides = pd.concat([rides_start, rides_end], ignore_index=True)
    else:
        rides = rides_start

    rides["pickup_datetime"] += timedelta(weeks=52)
    rides.sort_values(by=["pickup_location_id", "pickup_datetime"], inplace=True)

    return rides



def transform_ts_data_into_features(
    df: pd.DataFrame,
    feature_col: str = "rides",
    window_size: int = 12,
    step_size: int = 1
) -> pd.DataFrame:
    """
    Transforms Citi Bike time series data into tabular feature format for each station (location ID).
    Only features are generated (no target column). Features are lag values over a sliding window.

    Args:
        df (pd.DataFrame): Time series DataFrame with 'pickup_hour', 'pickup_location_id', and feature column.
        feature_col (str): Name of the feature column (default: 'rides').
        window_size (int): Number of hours to use as features per row (default: 12).
        step_size (int): Number of hours to shift the window (default: 1).

    Returns:
        pd.DataFrame: Tabular features including location_id and pickup_hour of prediction target.
    """
    location_ids = df["pickup_location_id"].unique()
    transformed_data = []

    for location_id in location_ids:
        try:
            location_data = df[df["pickup_location_id"] == location_id].reset_index(drop=True)
            values = location_data[feature_col].values
            times = location_data["pickup_hour"].values

            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one feature window.")

            rows = []
            for i in range(0, len(values) - window_size, step_size):
                feature_window = values[i: i + window_size]
                target_time = times[i + window_size]
                row = np.append(feature_window, [location_id, target_time])
                rows.append(row)

            feature_columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
            all_columns = feature_columns + ["pickup_location_id", "pickup_hour"]

            transformed_df = pd.DataFrame(rows, columns=all_columns)
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping pickup_location_id {location_id}: {e}")

    if not transformed_data:
        raise ValueError("No locations had enough data to create feature windows.")

    return pd.concat(transformed_data, ignore_index=True)

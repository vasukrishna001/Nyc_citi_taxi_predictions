from datetime import timedelta
from typing import Optional, Union

import pandas as pd
import plotly.express as px


def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: int,
    predictions: Optional[Union[pd.Series, pd.DataFrame]] = None,
):
    """
    Plots time series + actual and optional prediction for a selected row in the feature set.

    Args:
        features (pd.DataFrame): Feature DataFrame with 'pickup_hour' and 'pickup_location_id'.
        targets (pd.Series): Actual target values.
        row_id (int): Index to plot.
        predictions (Optional[Series/DataFrame]): Optional predicted values.

    Returns:
        plotly.graph_objects.Figure
    """
    location_features = features.iloc[row_id]
    actual_target = targets.iloc[row_id]

    # Get time-series lag columns
    time_series_columns = [col for col in features.columns if col.startswith("rides_t-")]
    time_series_values = [location_features[col] for col in time_series_columns] + [actual_target]

    # Generate corresponding timestamps
    pickup_time = location_features["pickup_hour"]
    time_series_dates = pd.date_range(
        end=pickup_time,
        periods=len(time_series_values),
        freq="h",
    )

    # Base plot
    fig = px.line(
        x=time_series_dates,
        y=time_series_values,
        template="plotly_white",
        markers=True,
        title=f"Pickup Hour: {pickup_time}, Location ID: {location_features['pickup_location_id']}",
        labels={"x": "Time", "y": "Ride Counts"},
    )

    # Add actual target point
    fig.add_scatter(
        x=[time_series_dates[-1]],
        y=[actual_target],
        line_color="green",
        mode="markers",
        marker_size=10,
        name="Actual Value",
    )

    # Optional predicted point
    if predictions is not None:
        #predicted_value = predictions.iloc[row_id] if isinstance(predictions, pd.Series) else predictions["predicted_demand"].iloc[row_id]
        if isinstance(predictions, pd.Series):
            predicted_value = predictions.iloc[row_id]
        elif isinstance(predictions, pd.DataFrame) and "predicted_demand" in predictions.columns:
            predicted_value = predictions["predicted_demand"].iloc[row_id]
        else:
            predicted_value = predictions[row_id]

        fig.add_scatter(
            x=[time_series_dates[-1]],
            y=[predicted_value],
            line_color="red",
            mode="markers",
            marker_symbol="x",
            marker_size=15,
            name="Prediction",
        )

    return fig


def plot_prediction(features: pd.DataFrame, prediction: pd.DataFrame):
    """
    Plots the full lag series plus prediction point.

    Args:
        features (pd.DataFrame): Feature row (usually 1 row DataFrame).
        prediction (pd.DataFrame): DataFrame with column 'predicted_demand'.

    Returns:
        plotly.graph_objects.Figure
    """
    time_series_columns = [col for col in features.columns if col.startswith("rides_t-")]
    lag_values = [features[col].iloc[0] for col in time_series_columns]
    predicted_value = prediction["predicted_demand"].iloc[0]

    time_series_values = lag_values + [predicted_value]

    pickup_hour = pd.Timestamp(features["pickup_hour"].iloc[0])
    time_series_dates = pd.date_range(
        end=pickup_hour,
        periods=len(time_series_values),
        freq="h",
    )

    df_plot = pd.DataFrame({
        "datetime": time_series_dates,
        "rides": time_series_values
    })

    fig = px.line(
        df_plot,
        x="datetime",
        y="rides",
        template="plotly_white",
        markers=True,
        title=f"Pickup Hour: {pickup_hour}, Location ID: {features['pickup_location_id'].iloc[0]}",
        labels={"datetime": "Time", "rides": "Ride Counts"},
    )

    fig.add_scatter(
        x=[pickup_hour],
        y=[predicted_value],
        line_color="red",
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        name="Prediction",
    )

    return fig

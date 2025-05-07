import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

# Function to calculate the average rides over the last 4 weeks (at 1-week intervals)
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    # Lags at weekly intervals (168 = 24 * 7 hours)
    week_hours = [7 * 24 * i for i in range(1, 5)]  # 168, 336, 504, 672
    columns = [f"rides_t-{lag}" for lag in week_hours]

    missing_cols = [col for col in columns if col not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing required lag columns: {missing_cols}")

    X = X.copy()
    X["average_rides_last_4_weeks"] = X[columns].mean(axis=1)
    return X


add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)


# Custom transformer to extract hour and day-of-week from timestamp
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek
        return X_.drop(columns=["pickup_hour", "pickup_location_id"])


add_temporal_features = TemporalFeatureEngineer()


# Create pipeline
def get_pipeline(**hyper_params):
    """
    Returns a pipeline that adds derived features and trains a LightGBM model.

    Parameters:
    ----------
    **hyper_params : dict
        Parameters to be passed to the LGBMRegressor.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        Full ML pipeline with feature engineering and regression model.
    """
    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyper_params),
    )
    return pipeline

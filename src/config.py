import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Define directories
PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PARENT_DIR / "models"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Hopsworks configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# Feature store configuration
FEATURE_GROUP_NAME = "citi_bike_hourly_features"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "citi_bike_hourly_feature_view"
FEATURE_VIEW_VERSION = 1

# Model configuration
MODEL_NAME = "citi_bike_ride_predictor_next_hour"
MODEL_VERSION = 1

# Prediction storage configuration
FEATURE_GROUP_MODEL_PREDICTION = "citi_bike_hourly_model_prediction"

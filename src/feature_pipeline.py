import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd

import src.config as config
from src.data_utils import fetch_batch_raw_data, transform_raw_data_into_ts_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Step 1: Get the current UTC time (timezone-aware → naive UTC)
current_date = pd.to_datetime(datetime.now(timezone.utc)).ceil("h").tz_convert(None)
logger.info(f"Current date and time (UTC): {current_date}")

# Step 2: Define data fetching range
fetch_data_to = current_date
fetch_data_from = current_date - timedelta(days=28)
logger.info(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# Step 3: Fetch raw data
logger.info("Fetching raw data...")
rides = fetch_batch_raw_data(fetch_data_from, fetch_data_to)
logger.info(f"Raw data fetched. Number of records: {len(rides)}")

# Step 4: Transform raw data into time-series data
logger.info("Transforming raw data into time-series format...")
ts_data = transform_raw_data_into_ts_data(rides)
logger.info(f"Transformation complete. Records in time-series data: {len(ts_data)}")

# Optional: Add partition column (hourly)
ts_data["event_hour"] = ts_data["pickup_hour"]

# Optional: Preview sample for debugging
logger.info(f"Sample data:\n{ts_data.head(2)}")
logger.info(f"Schema:\n{ts_data.dtypes}")

# Step 5: Connect to Hopsworks
logger.info("Logging into Hopsworks...")
project = hopsworks.login(
    project=config.HOPSWORKS_PROJECT_NAME,
    api_key_value=config.HOPSWORKS_API_KEY,
)
logger.info("Successfully connected to Hopsworks project.")

# Step 6: Connect to feature store
logger.info("Connecting to feature store...")
feature_store = project.get_feature_store()
logger.info("Feature store ready.")

# Step 7: Get feature group
logger.info(
    f"Getting feature group: {config.FEATURE_GROUP_NAME} (v{config.FEATURE_GROUP_VERSION})"
)
feature_group = feature_store.get_feature_group(
    name=config.FEATURE_GROUP_NAME,
    version=config.FEATURE_GROUP_VERSION,
)

# Step 8: Insert time-series data
logger.info("Inserting data into feature group...")
feature_group.insert(
    ts_data,
    write_options={
        "wait_for_job": False,
        "partition_key": ["event_hour"]  # Optional: enable if using partitioning
    }
)
logger.info("✅ Data insertion complete.")

import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/spotify_churn_dataset.csv"
PROCESSED_DIR = "data/processed"
TARGET = "ads_listened_per_week"

CATEGORICAL_FEATURES = ["gender", "country", "subscription_type", "device_type"]
NUMERICAL_FEATURES = ["age", "songs_played_per_day", "skip_rate", "listening_time", "offline_listening"]


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Raw dataset: %d rows, %d columns", len(df), len(df.columns))
    df = df.drop(columns=["user_id", "is_churned"])
    logger.info("Dropped columns [user_id, is_churned] — %d features remaining", len(df.columns) - 1)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def upload_to_azure(local_path: str, blob_name: str):
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("AZURE_CONTAINER_NAME", "mlops-spotify")

    if not conn_str:
        logger.debug("Azure upload skipped (AZURE_STORAGE_CONNECTION_STRING not set)")
        return

    try:
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(conn_str)
        container_client = client.get_container_client(container)

        if not container_client.exists():
            container_client.create_container()
            logger.info("Created Azure container '%s'", container)

        with open(local_path, "rb") as f:
            container_client.upload_blob(name=blob_name, data=f, overwrite=True)

        logger.info("Uploaded %s to Azure container '%s'", blob_name, container)
    except Exception as e:
        logger.warning("Azure upload failed for %s: %s", blob_name, e)


def preprocess(random_state: int = 42):
    t0 = time.time()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logger.info("Starting preprocessing (random_state=%d)", random_state)

    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Split 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )
    logger.info(
        "Split 70/15/15 — train=%d, val=%d, test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    # Fit preprocessor on train only (no data leakage)
    logger.info("Fitting preprocessor on train set only")
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)
    logger.info("Output shape after encoding: %s", X_train_p.shape)

    # Save Parquet splits (raw, for retraining and traceability)
    splits = [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]
    for name, X_split, y_split in splits:
        df_split = X_split.copy()
        df_split[TARGET] = y_split.values
        path = f"{PROCESSED_DIR}/{name}.parquet"
        df_split.to_parquet(path, index=False)
        logger.info("Saved %s.parquet (%d rows)", name, len(df_split))
        upload_to_azure(path, f"processed/{name}.parquet")

    # Save numpy arrays for fast training
    arrays = {
        "X_train": X_train_p, "X_val": X_val_p, "X_test": X_test_p,
        "y_train": y_train.values, "y_val": y_val.values, "y_test": y_test.values,
    }
    for name, arr in arrays.items():
        np.save(f"{PROCESSED_DIR}/{name}.npy", arr)
    logger.info("Saved numpy arrays to %s/", PROCESSED_DIR)

    joblib.dump(preprocessor, f"{PROCESSED_DIR}/preprocessor.pkl")
    logger.info("Saved preprocessor.pkl")

    elapsed = time.time() - t0
    logger.info("Preprocessing done in %.1fs", elapsed)


if __name__ == "__main__":
    preprocess()

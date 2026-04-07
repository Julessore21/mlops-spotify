import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

RAW_DATA_PATH = "data/raw/spotify_churn_dataset.csv"
PROCESSED_DIR = "data/processed"
TARGET = "listening_time"

CATEGORICAL_FEATURES = ["gender", "country", "subscription_type", "device_type"]
NUMERICAL_FEATURES = ["age", "songs_played_per_day", "skip_rate", "ads_listened_per_week", "offline_listening"]


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["user_id", "is_churned"])
    return df


def build_preprocessor() -> ColumnTransformer:
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def preprocess(test_size: float = 0.2, random_state: int = 42):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_data()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    np.save(f"{PROCESSED_DIR}/X_train.npy", X_train_processed)
    np.save(f"{PROCESSED_DIR}/X_test.npy", X_test_processed)
    np.save(f"{PROCESSED_DIR}/y_train.npy", y_train.values)
    np.save(f"{PROCESSED_DIR}/y_test.npy", y_test.values)
    joblib.dump(preprocessor, f"{PROCESSED_DIR}/preprocessor.pkl")

    print(f"Train size: {X_train_processed.shape}")
    print(f"Test size:  {X_test_processed.shape}")
    print("Preprocessing done. Files saved in data/processed/")


if __name__ == "__main__":
    preprocess()

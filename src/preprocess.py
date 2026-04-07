import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def preprocess(random_state: int = 42):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_data()

    # Split 70 / 15 / 15
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )

    # Fit preprocessor on train only
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    # Save as Parquet (raw splits) for traceability
    train_df = X_train.copy(); train_df[TARGET] = y_train.values
    val_df   = X_val.copy();   val_df[TARGET]   = y_val.values
    test_df  = X_test.copy();  test_df[TARGET]  = y_test.values

    train_df.to_parquet(f"{PROCESSED_DIR}/train.parquet", index=False)
    val_df.to_parquet(f"{PROCESSED_DIR}/val.parquet",     index=False)
    test_df.to_parquet(f"{PROCESSED_DIR}/test.parquet",   index=False)

    # Save numpy arrays for training
    np.save(f"{PROCESSED_DIR}/X_train.npy", X_train_p)
    np.save(f"{PROCESSED_DIR}/X_val.npy",   X_val_p)
    np.save(f"{PROCESSED_DIR}/X_test.npy",  X_test_p)
    np.save(f"{PROCESSED_DIR}/y_train.npy", y_train.values)
    np.save(f"{PROCESSED_DIR}/y_val.npy",   y_val.values)
    np.save(f"{PROCESSED_DIR}/y_test.npy",  y_test.values)

    joblib.dump(preprocessor, f"{PROCESSED_DIR}/preprocessor.pkl")

    print(f"Train : {X_train_p.shape} ({len(X_train_p)/len(df)*100:.0f}%)")
    print(f"Val   : {X_val_p.shape}   ({len(X_val_p)/len(df)*100:.0f}%)")
    print(f"Test  : {X_test_p.shape}  ({len(X_test_p)/len(df)*100:.0f}%)")
    print("Preprocessing done. Files saved in data/processed/")


if __name__ == "__main__":
    preprocess()

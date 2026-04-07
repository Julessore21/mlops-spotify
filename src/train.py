import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
EXPERIMENT_NAME = "spotify-listening-time-prediction"


def load_processed_data():
    X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(fit_intercept: bool = True):
    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_processed_data()

    with mlflow.start_run():
        params = {"fit_intercept": fit_intercept}
        mlflow.log_params(params)

        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = compute_metrics(y_train, y_pred_train)
        test_metrics = compute_metrics(y_test, y_pred_test)

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        mlflow.sklearn.log_model(model, artifact_path="linear_regression_model")

        model_path = f"{MODEL_DIR}/linear_regression.pkl"
        joblib.dump(model, model_path)

        print("=== Train metrics ===")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("=== Test metrics ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nModel saved to {model_path}")
        print(f"MLflow run logged under experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    train()

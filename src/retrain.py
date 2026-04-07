"""
Retraining pipeline — simulates arrival of new data.

Usage:
    python src/retrain.py --batch val    # use val.parquet as new data
    python src/retrain.py --batch test   # use test.parquet as new data
"""
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
EXPERIMENT_NAME = "spotify-ads-prediction"
TARGET = "ads_listened_per_week"

MODELS_GRID = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {"fit_intercept": [True, False]},
    },
    "Ridge": {
        "model": Ridge(),
        "params": {"alpha": [0.1, 1.0, 10.0, 100.0], "fit_intercept": [True, False]},
    },
    "Lasso": {
        "model": Lasso(max_iter=5000),
        "params": {"alpha": [0.01, 0.1, 1.0, 10.0], "fit_intercept": [True, False]},
    },
}


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def retrain(batch: str = "val"):
    """
    Merge original train split with new batch data, retrain all models,
    save best model and update the API-facing artifact.
    """
    assert batch in ("val", "test"), "batch must be 'val' or 'test'"

    preprocessor = joblib.load(f"{PROCESSED_DIR}/preprocessor.pkl")

    # Load original train
    X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")

    # Load new batch and preprocess
    new_df = pd.read_parquet(f"{PROCESSED_DIR}/{batch}.parquet")
    X_new = preprocessor.transform(new_df.drop(columns=[TARGET]))
    y_new = new_df[TARGET].values

    # Merge
    X_combined = np.vstack([X_train, X_new])
    y_combined = np.concatenate([y_train, y_new])

    # Evaluate on the other split as held-out test
    held_out = "test" if batch == "val" else "val"
    X_held = np.load(f"{PROCESSED_DIR}/X_{held_out}.npy")
    y_held = np.load(f"{PROCESSED_DIR}/y_{held_out}.npy")

    print(f"\nRetraining on train + {batch} ({len(X_combined)} samples)")
    print(f"Evaluating on {held_out} ({len(X_held)} samples)\n")

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_name, best_model, best_r2 = None, None, -float("inf")

    for name, config in MODELS_GRID.items():
        with mlflow.start_run(run_name=f"{name}_retrain_{batch}"):
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("retrain_batch", batch)

            grid = GridSearchCV(config["model"], config["params"], cv=5, scoring="r2", n_jobs=-1)
            grid.fit(X_combined, y_combined)
            model = grid.best_estimator_

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("cv_best_r2", float(grid.best_score_))
            mlflow.log_metric("train_size", len(X_combined))

            metrics = compute_metrics(y_held, model.predict(X_held))
            mlflow.log_metrics({f"{held_out}_{k}": v for k, v in metrics.items()})
            mlflow.sklearn.log_model(model, artifact_path=f"{name}_retrain_model")

            print(f"  {name} — CV R²: {grid.best_score_:.4f} | {held_out} R²: {metrics['r2']:.4f}")

            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_name = name
                best_model = model

    model_path = f"{MODEL_DIR}/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nBest model: {best_name} (R²: {best_r2:.4f}) → saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, default="val", choices=["val", "test"],
                        help="New data batch to incorporate: 'val' or 'test'")
    args = parser.parse_args()
    retrain(batch=args.batch)

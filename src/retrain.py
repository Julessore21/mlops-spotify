"""
Retraining pipeline — simulates arrival of new data.

Usage:
    python src/retrain.py --batch val    # use val.parquet as new data
    python src/retrain.py --batch test   # use test.parquet as new data
"""
import argparse
import logging
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    """Merge original train split with new batch data, retrain, save best model."""
    assert batch in ("val", "test"), "batch must be 'val' or 'test'"

    t0 = time.time()
    logger.info("Starting retrain pipeline — new data batch: '%s'", batch)

    preprocessor = joblib.load(f"{PROCESSED_DIR}/preprocessor.pkl")

    X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
    logger.info("Original train set: %d samples", len(X_train))

    new_df = pd.read_parquet(f"{PROCESSED_DIR}/{batch}.parquet")
    X_new = preprocessor.transform(new_df.drop(columns=[TARGET]))
    y_new = new_df[TARGET].values
    logger.info("New batch (%s): %d samples", batch, len(X_new))

    X_combined = np.vstack([X_train, X_new])
    y_combined = np.concatenate([y_train, y_new])
    logger.info("Combined training set: %d samples", len(X_combined))

    held_out = "test" if batch == "val" else "val"
    X_held = np.load(f"{PROCESSED_DIR}/X_{held_out}.npy")
    y_held = np.load(f"{PROCESSED_DIR}/y_{held_out}.npy")
    logger.info("Held-out evaluation set: '%s' (%d samples)", held_out, len(X_held))

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_name, best_model, best_r2 = None, None, -float("inf")

    for name, config in MODELS_GRID.items():
        logger.info("Retraining %s...", name)
        t_model = time.time()

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
            signature = infer_signature(X_combined, model.predict(X_combined))
            mlflow.sklearn.log_model(
                model,
                artifact_path=f"{name}_retrain_model",
                signature=signature,
                input_example=X_combined[:3],
            )

            logger.info(
                "%s — CV R²: %.4f | %s R²: %.4f | params: %s | %.1fs",
                name, grid.best_score_, held_out, metrics["r2"],
                grid.best_params_, time.time() - t_model,
            )

            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_name = name
                best_model = model

    model_path = f"{MODEL_DIR}/best_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(
        "Best model: %s (%s R²=%.4f) saved to %s",
        best_name, held_out, best_r2, model_path,
    )
    logger.info("Retrain done in %.1fs — call POST /reload to update the API", time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch", type=str, default="val", choices=["val", "test"],
        help="New data batch to incorporate: 'val' or 'test'",
    )
    args = parser.parse_args()
    retrain(batch=args.batch)

import logging
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
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


def load_processed_data():
    logger.info("Loading processed data from %s/", PROCESSED_DIR)
    X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
    X_val = np.load(f"{PROCESSED_DIR}/X_val.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
    y_val = np.load(f"{PROCESSED_DIR}/y_val.npy")
    logger.info("Train: %s | Val: %s", X_train.shape, X_val.shape)
    return X_train, X_val, y_train, y_val


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_with_gridsearch(name, model, param_grid, X_train, X_val, y_train, y_val):
    logger.info("Training %s with GridSearchCV...", name)
    t0 = time.time()

    with mlflow.start_run(run_name=f"{name}_gridsearch"):
        mlflow.set_tag("model_type", name)

        grid = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_r2", float(grid.best_score_))

        train_metrics = compute_metrics(y_train, best_model.predict(X_train))
        val_metrics = compute_metrics(y_val, best_model.predict(X_val))

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            best_model,
            artifact_path=f"{name}_best_model",
            signature=signature,
            input_example=X_train[:3],
        )

        elapsed = time.time() - t0
        logger.info(
            "%s — CV R²: %.4f | Val R²: %.4f | Val RMSE: %.4f | params: %s | %.1fs",
            name, grid.best_score_, val_metrics["r2"], val_metrics["rmse"], best_params, elapsed,
        )

    return best_model, val_metrics


def train():
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("MLflow experiment: '%s'", EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_val, y_train, y_val = load_processed_data()

    best_name, best_model, best_r2 = None, None, -float("inf")

    for name, config in MODELS_GRID.items():
        trained_model, metrics = train_with_gridsearch(
            name, config["model"], config["params"],
            X_train, X_val, y_train, y_val,
        )
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_name = name
            best_model = trained_model

    model_path = f"{MODEL_DIR}/best_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(
        "Best model: %s (val R²=%.4f) saved to %s",
        best_name, best_r2, model_path,
    )
    logger.info("Training done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    train()

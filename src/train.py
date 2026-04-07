import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
EXPERIMENT_NAME = "spotify-listening-time-prediction"

MODELS_GRID = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {
            "fit_intercept": [True, False],
        },
    },
    "Ridge": {
        "model": Ridge(),
        "params": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
            "fit_intercept": [True, False],
        },
    },
    "Lasso": {
        "model": Lasso(max_iter=5000),
        "params": {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "fit_intercept": [True, False],
        },
    },
}


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


def train_with_gridsearch(name, model, param_grid, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=f"{name}_gridsearch"):
        mlflow.set_tag("model_type", name)

        grid = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_r2", float(grid.best_score_))

        train_metrics = compute_metrics(y_train, best_model.predict(X_train))
        test_metrics = compute_metrics(y_test, best_model.predict(X_test))

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        mlflow.sklearn.log_model(best_model, artifact_path=f"{name}_best_model")

        print(f"\n=== {name} (GridSearch) ===")
        print(f"  Best params : {best_params}")
        print(f"  CV R²       : {grid.best_score_:.4f}")
        print(f"  Test R²     : {test_metrics['r2']:.4f} | Test RMSE: {test_metrics['rmse']:.4f}")

    return best_model, test_metrics


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_processed_data()

    best_name, best_model, best_r2 = None, None, -float("inf")

    for name, config in MODELS_GRID.items():
        trained_model, metrics = train_with_gridsearch(
            name, config["model"], config["params"],
            X_train, X_test, y_train, y_test
        )
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_name = name
            best_model = trained_model

    model_path = f"{MODEL_DIR}/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nBest model: {best_name} (test R²: {best_r2:.4f}) → saved to {model_path}")


if __name__ == "__main__":
    train()

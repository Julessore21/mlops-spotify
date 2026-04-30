import logging
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
EXPERIMENT_NAME = "spotify-ads-prediction"


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate():
    model_path = f"{MODEL_DIR}/best_model.pkl"
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading test set from %s/", PROCESSED_DIR)
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    logger.info("Test set: %d samples", len(y_test))

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    logger.info("Test R²   = %.4f", metrics["r2"])
    logger.info("Test RMSE = %.4f", metrics["rmse"])
    logger.info("Test MAE  = %.4f", metrics["mae"])

    logger.info("Logging final metrics to MLflow experiment '%s'", EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="final_evaluation"):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    # Plots
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.4, edgecolors="k", linewidths=0.3)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0].set_xlabel("Valeurs réelles (ads/semaine)")
    axes[0].set_ylabel("Valeurs prédites (ads/semaine)")
    axes[0].set_title("Réel vs Prédit")

    axes[1].scatter(y_pred, residuals, alpha=0.4, edgecolors="k", linewidths=0.3)
    axes[1].axhline(0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Valeurs prédites (ads/semaine)")
    axes[1].set_ylabel("Résidus")
    axes[1].set_title("Graphique des résidus")

    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plot_path = "reports/evaluation_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Plots saved to %s", plot_path)


if __name__ == "__main__":
    evaluate()

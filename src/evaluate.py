import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def evaluate():
    model = joblib.load(f"{MODEL_DIR}/linear_regression.pkl")
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")

    y_pred = model.predict(X_test)

    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.4, edgecolors="k", linewidths=0.3)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0].set_xlabel("Actual listening_time")
    axes[0].set_ylabel("Predicted listening_time")
    axes[0].set_title("Actual vs Predicted")

    axes[1].scatter(y_pred, residuals, alpha=0.4, edgecolors="k", linewidths=0.3)
    axes[1].axhline(0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted listening_time")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals Plot")

    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/evaluation_plots.png", dpi=150)
    plt.show()
    print("Plot saved to reports/evaluation_plots.png")


if __name__ == "__main__":
    evaluate()

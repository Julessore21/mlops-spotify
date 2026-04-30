import logging
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "data/processed/preprocessor.pkl"

model = None
preprocessor = None


def _load_artifacts():
    global model, preprocessor
    logger.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    logger.info("Loading preprocessor from %s", PREPROCESSOR_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info("Artifacts loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield
    logger.info("API shutting down")


app = FastAPI(title="Spotify Ads Prediction API", lifespan=lifespan)


class UserFeatures(BaseModel):
    gender: str
    age: int
    country: str
    subscription_type: str
    songs_played_per_day: int
    skip_rate: float
    device_type: str
    listening_time: float
    offline_listening: int


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000
    logger.info("%s %s → %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/")
def root():
    return {"message": "Spotify Ads Prediction API"}


@app.post("/predict")
def predict(features: UserFeatures):
    try:
        df = pd.DataFrame([features.model_dump()])
        X = preprocessor.transform(df)
        prediction = float(model.predict(X)[0])
        result = round(prediction, 2)
        logger.info("Prediction: input=%s → %.2f ads/week", features.model_dump(), result)
        return {"predicted_ads_listened_per_week": result}
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction error")


@app.post("/reload")
def reload_model():
    """Recharge le modèle depuis le disque après réentraînement."""
    try:
        _load_artifacts()
        logger.info("Model reloaded successfully via /reload endpoint")
        return {"status": "model reloaded"}
    except Exception as e:
        logger.error("Reload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Reload error")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

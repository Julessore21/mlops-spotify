from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Spotify Listening Time Predictor")

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "data/processed/preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


class UserFeatures(BaseModel):
    gender: str
    age: int
    country: str
    subscription_type: str
    songs_played_per_day: int
    skip_rate: float
    device_type: str
    ads_listened_per_week: int
    offline_listening: int


@app.get("/")
def root():
    return {"message": "Spotify Listening Time Predictor API"}


@app.post("/predict")
def predict(features: UserFeatures):
    df = pd.DataFrame([features.model_dump()])
    X = preprocessor.transform(df)
    prediction = model.predict(X)[0]
    return {"predicted_listening_time": round(float(prediction), 2)}


@app.get("/health")
def health():
    return {"status": "ok"}

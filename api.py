
from contextlib import asynccontextmanager

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model_io import load_model, predict as model_predict


# =========================================================
# Global model container
# =========================================================

MODEL = {}


# =========================================================
# Application lifespan
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        MODEL["data"] = load_model()

        print("Model loaded successfully")

    except FileNotFoundError:
        print(" Model file not found")

    yield

    MODEL.clear()


# =========================================================
# FastAPI app
# =========================================================

app = FastAPI(
    title="Rain Prediction API",
    description="Predict whether it will rain tomorrow.",
    version="2.0.0",
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Input schema
# =========================================================

class WeatherInput(BaseModel):

    Humidity3pm: float = Field(..., example=55.0)
    Humidity9am: float = Field(..., example=80.0)

    Rainfall: float = Field(..., example=5.0)

    Sunshine: float = Field(..., example=8.0)

    Cloud3pm: float = Field(..., example=4.0)
    Cloud9am: float = Field(..., example=5.0)

    Pressure9am: float = Field(..., example=1012.0)

    WindGustSpeed: float = Field(..., example=40.0)

    Temp3pm: float = Field(..., example=22.0)

    RainToday: int = Field(..., example=0)


# =========================================================
# Output schema
# =========================================================

class PredictionOutput(BaseModel):

    prediction: str
    probability: float
    rain: bool


# =========================================================
# Routes
# =========================================================

@app.get("/", tags=["General"])
def root():

    return {
        "message": "Rain Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", tags=["General"])
def health():

    model_loaded = "data" in MODEL

    return {
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "model_name": (
            MODEL["data"].get("model_name")
            if model_loaded
            else None
        ),
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
)
def predict(weather: WeatherInput):

    if "data" not in MODEL:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    input_df = pd.DataFrame([weather.model_dump()])

    label, probability = model_predict(
        MODEL["data"],
        input_df,
    )

    return PredictionOutput(
        prediction=label,
        probability=probability,
        rain=probability >= 0.5,
    )
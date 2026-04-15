from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("model.joblib")

app = FastAPI(
    title="ML FastAPI Docker Demo",
    description="A simple Iris classifier served with FastAPI.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1, gt=0)
    sepal_width: float = Field(..., example=3.5, gt=0)
    petal_length: float = Field(..., example=1.4, gt=0)
    petal_width: float = Field(..., example=0.2, gt=0)


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str


def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.joblib was not found. Run `python3 train.py` before starting the API."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "ML API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        artifact = load_artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    model = artifact["model"]
    target_names = artifact["target_names"]

    features = np.array(
        [
            [
                request.sepal_length,
                request.sepal_width,
                request.petal_length,
                request.petal_width,
            ]
        ]
    )

    predicted_class = int(model.predict(features)[0])
    predicted_label = target_names[predicted_class]

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=predicted_label,
    )

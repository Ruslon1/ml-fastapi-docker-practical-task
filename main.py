from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_service import predict_species

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


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "ML API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        result = predict_species(
            [
                request.sepal_length,
                request.sepal_width,
                request.petal_length,
                request.petal_width,
            ]
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        predicted_class=result["predicted_class"],
        predicted_label=result["predicted_label"],
    )

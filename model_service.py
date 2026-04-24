from pathlib import Path
from typing import Any

import joblib
import numpy as np


MODEL_PATH = Path("model.joblib")
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def load_artifact(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} was not found. Run `python3 train.py` before starting the app."
        )
    return joblib.load(model_path)


def predict_species(features: list[float], model_path: Path = MODEL_PATH) -> dict[str, Any]:
    artifact = load_artifact(model_path)
    model = artifact["model"]
    target_names = artifact["target_names"]
    feature_array = np.array([features])

    predicted_class = int(model.predict(feature_array)[0])

    return {
        "predicted_class": predicted_class,
        "predicted_label": target_names[predicted_class],
    }

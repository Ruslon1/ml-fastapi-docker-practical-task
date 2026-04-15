from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path("model.joblib")


def main() -> None:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    artifact = {
        "model": model,
        "target_names": iris.target_names.tolist(),
        "feature_names": iris.feature_names,
        "accuracy": accuracy,
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH.resolve()}")
    print(f"Validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

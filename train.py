from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path("model.joblib")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "iris-classifier"
REGISTERED_MODEL_NAME = "iris-logistic-regression"
SPLIT_RANDOM_STATE = 42
TEST_SIZE = 0.2
CLASSIFIER_MAX_ITER = 300
CLASSIFIER_RANDOM_STATE = 42


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=CLASSIFIER_MAX_ITER,
                    random_state=CLASSIFIER_RANDOM_STATE,
                ),
            ),
        ]
    )


def training_params() -> dict[str, int | float | str]:
    return {
        "model_type": "LogisticRegression",
        "classifier_max_iter": CLASSIFIER_MAX_ITER,
        "classifier_random_state": CLASSIFIER_RANDOM_STATE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "test_size": TEST_SIZE,
    }


def evaluate_model(y_test, predictions) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, average="weighted"),
    }


def save_model_artifact(model: Pipeline, target_names: list[str], metrics: dict[str, float]) -> None:
    artifact = {
        "model": model,
        "target_names": target_names,
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_PATH)


def log_with_mlflow(model: Pipeline, params: dict[str, int | float | str], metrics: dict[str, float]) -> str:
    import os

    import mlflow
    import mlflow.sklearn
    import numpy as np

    os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "false"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    input_example = np.array([[5.1, 3.5, 1.4, 0.2]])

    with mlflow.start_run(run_name="logistic-regression") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="saved_model")

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
        )
        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=REGISTERED_MODEL_NAME,
        )

    return f"{REGISTERED_MODEL_NAME} v{model_version.version} (run {run.info.run_id})"


def main() -> None:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=iris.target,
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = evaluate_model(y_test, predictions)
    params = training_params()

    save_model_artifact(model, iris.target_names.tolist(), metrics)
    registered_model = log_with_mlflow(model, params, metrics)

    print(f"Model saved to {MODEL_PATH.resolve()}")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1-score: {metrics['f1_score']:.4f}")
    print(f"Registered model: {registered_model}")


if __name__ == "__main__":
    main()

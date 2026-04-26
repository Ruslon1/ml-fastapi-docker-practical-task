from argparse import ArgumentParser
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
DEFAULT_EXPERIMENT_NAME = "iris-model-comparison"
SPLIT_RANDOM_STATE = 42
TEST_SIZE = 0.2
LOGISTIC_PARAMS = {
    "classifier_max_iter": 300,
    "classifier_random_state": 42,
}
CATBOOST_PARAMS = {
    "iterations": 200,
    "depth": 4,
    "learning_rate": 0.05,
    "loss_function": "MultiClass",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


def build_model(model_name: str):
    if model_name == "logistic":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=LOGISTIC_PARAMS["classifier_max_iter"],
                        random_state=LOGISTIC_PARAMS["classifier_random_state"],
                    ),
                ),
            ]
        )

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**CATBOOST_PARAMS)

    raise ValueError(f"Unknown model: {model_name}")


def training_params(model_name: str) -> dict[str, int | float | str]:
    params = {
        "model_name": model_name,
        "split_random_state": SPLIT_RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    if model_name == "logistic":
        params.update(LOGISTIC_PARAMS)

    if model_name == "catboost":
        params.update(CATBOOST_PARAMS)

    return params


def evaluate_model(y_test, predictions) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, average="weighted"),
    }


def save_model_artifact(model, target_names: list[str], metrics: dict[str, float]) -> None:
    artifact = {
        "model": model,
        "target_names": target_names,
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_PATH)


def log_with_mlflow(
    model,
    model_name: str,
    experiment_name: str,
    registered_model_name: str,
    params: dict[str, int | float | str],
    metrics: dict[str, float],
) -> str:
    import os

    import mlflow
    import numpy as np

    os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "false"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    input_example = np.array([[5.1, 3.5, 1.4, 0.2]])

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="saved_model")

        if model_name == "catboost":
            import mlflow.catboost

            model_info = mlflow.catboost.log_model(
                cb_model=model,
                name="model",
                input_example=input_example,
            )
        else:
            import mlflow.sklearn

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=input_example,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            )

        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=registered_model_name,
        )

    return f"{registered_model_name} v{model_version.version} (run {run.info.run_id})"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["logistic", "catboost"],
        default="logistic",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
    )
    parser.add_argument(
        "--registered-model-name",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registered_model_name = args.registered_model_name or f"iris-{args.model}-classifier"

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=iris.target,
    )

    model = build_model(args.model)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = evaluate_model(y_test, predictions)
    params = training_params(args.model)

    save_model_artifact(model, iris.target_names.tolist(), metrics)

    registered_model = log_with_mlflow(
        model=model,
        model_name=args.model,
        experiment_name=args.experiment_name,
        registered_model_name=registered_model_name,
        params=params,
        metrics=metrics,
    )

    print(f"Model saved to {MODEL_PATH.resolve()}")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1-score: {metrics['f1_score']:.4f}")
    print(f"Registered model: {registered_model}")


if __name__ == "__main__":
    main()

from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from deep_model import IrisNet, TorchIrisClassifier


MODEL_PATH = Path("model.joblib")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "iris-model-comparison"
DEFAULT_REGISTERED_MODEL_NAME = "iris-pytorch-mlp-classifier"
SPLIT_RANDOM_STATE = 42
TEST_SIZE = 0.2
TORCH_SEED = 42
HIDDEN_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 0.01


def training_params() -> dict[str, int | float | str]:
    return {
        "model_name": "pytorch-mlp",
        "hidden_size": HIDDEN_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "test_size": TEST_SIZE,
        "torch_seed": TORCH_SEED,
    }


def train_model(x_train, y_train) -> tuple[IrisNet, StandardScaler]:
    torch.manual_seed(TORCH_SEED)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)

    features = torch.tensor(x_train_scaled, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.long)

    model = IrisNet(hidden_size=HIDDEN_SIZE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

    return model.eval(), scaler


def evaluate_model(classifier: TorchIrisClassifier, x_test, y_test) -> dict[str, float]:
    predictions = classifier.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, average="weighted"),
    }


def save_model_artifact(
    classifier: TorchIrisClassifier,
    target_names: list[str],
    metrics: dict[str, float],
) -> None:
    artifact = {
        "model": classifier,
        "target_names": target_names,
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_PATH)


def log_with_mlflow(
    model: IrisNet,
    experiment_name: str,
    registered_model_name: str,
    params: dict[str, int | float | str],
    metrics: dict[str, float],
) -> str:
    import os

    import mlflow
    import mlflow.pytorch

    os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "false"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    input_example = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    with mlflow.start_run(run_name="pytorch-mlp") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="saved_model")

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            input_example=input_example,
        )
        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=registered_model_name,
        )

    return f"{registered_model_name} v{model_version.version} (run {run.info.run_id})"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
    )
    parser.add_argument(
        "--registered-model-name",
        default=DEFAULT_REGISTERED_MODEL_NAME,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=iris.target,
    )

    model, scaler = train_model(x_train, y_train)
    classifier = TorchIrisClassifier(model, scaler)
    metrics = evaluate_model(classifier, x_test, y_test)
    params = training_params()

    save_model_artifact(classifier, iris.target_names.tolist(), metrics)
    registered_model = log_with_mlflow(
        model=model,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        params=params,
        metrics=metrics,
    )

    print(f"Model saved to {MODEL_PATH.resolve()}")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1-score: {metrics['f1_score']:.4f}")
    print(f"Registered model: {registered_model}")


if __name__ == "__main__":
    main()

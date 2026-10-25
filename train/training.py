import os

import bentoml
import mlflow
import mlflow.sklearn
from prefect import flow, task
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def fit_classifier(
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, float]:
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    iris = load_iris()
    features_train, features_test, labels_train, labels_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=random_state,
        stratify=iris.target,
    )
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(features_train, labels_train)
    accuracy = accuracy_score(labels_test, model.predict(features_test))
    return model, accuracy


@task(name="train-model")
def train_model(
    n_estimators: int,
    test_size: float,
    random_state: int,
) -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("iris-classification")

    model, accuracy = fit_classifier(
        n_estimators=n_estimators,
        test_size=test_size,
        random_state=random_state,
    )
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "test_size": test_size,
                "random_state": random_state,
            }
        )
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            model,
            name="iris_classifier",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    bento_model = bentoml.sklearn.save_model(
        "iris_classifier",
        model,
        metadata={"accuracy": accuracy, "mlflow_run_id": run.info.run_id},
    )
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"MLflow run: {run.info.run_id}")
    print(f"BentoML model: {bento_model.tag}")
    return str(bento_model.tag)


@flow(name="iris-training-pipeline", log_prints=True)
def training_pipeline(
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    return train_model(
        n_estimators=n_estimators,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    training_pipeline()
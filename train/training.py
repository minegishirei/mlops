import os

import bentoml
import mlflow
import mlflow.sklearn
from prefect import flow, task
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def fit_classifier() -> tuple[RandomForestClassifier, float]:
    iris = load_iris()
    features_train, features_test, labels_train, labels_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_train, labels_train)
    accuracy = accuracy_score(labels_test, model.predict(features_test))
    return model, accuracy


@task(name="train-and-register-model")
def train_and_register_model() -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("iris-classification")

    model, accuracy = fit_classifier()
    with mlflow.start_run() as run:
        mlflow.log_params({"model": "RandomForestClassifier", "n_estimators": 100})
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
def training_pipeline() -> str:
    return train_and_register_model()


if __name__ == "__main__":
    training_pipeline()
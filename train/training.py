import os
import time

import bentoml
import mlflow
import mlflow.sklearn
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode
from prefect import flow, task
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from telemetry import configure_telemetry

configure_telemetry("mlops-training")
tracer = trace.get_tracer("mlops.training")
meter = metrics.get_meter("mlops.training")
training_runs = meter.create_counter("mlops.training.runs", unit="{run}")
training_failures = meter.create_counter("mlops.training.failures", unit="{failure}")
training_duration = meter.create_histogram("mlops.training.duration", unit="s")


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


@task(name="train-and-register-model")
def train_and_register_model(
    n_estimators: int,
    test_size: float,
    random_state: int,
) -> str:
    started_at = time.perf_counter()
    metric_attributes = {"flow.name": "iris-training-pipeline"}
    with tracer.start_as_current_span("training.train-and-register-model") as span:
        span.set_attributes(
            {
                "mlops.model.family": "RandomForestClassifier",
                "mlops.training.n_estimators": n_estimators,
                "mlops.training.test_size": test_size,
                "mlops.training.random_state": random_state,
            }
        )
        try:
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
                span_context = span.get_span_context()
                if span_context.is_valid:
                    mlflow.set_tag("otel.trace_id", f"{span_context.trace_id:032x}")
                    mlflow.set_tag("otel.span_id", f"{span_context.span_id:016x}")
                mlflow.sklearn.log_model(
                    model,
                    name="iris_classifier",
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )

                bento_model = bentoml.sklearn.save_model(
                    "iris_classifier",
                    model,
                    metadata={
                        "accuracy": accuracy,
                        "mlflow_run_id": run.info.run_id,
                        "training_trace_id": f"{span_context.trace_id:032x}"
                        if span_context.is_valid
                        else "",
                    },
                )
                mlflow.set_tags(
                    {
                        "bentoml.model_tag": str(bento_model.tag),
                        "otel.trace_id": f"{span_context.trace_id:032x}"
                        if span_context.is_valid
                        else "",
                    }
                )

            span.set_attributes(
                {
                    "mlflow.run_id": run.info.run_id,
                    "mlops.model.tag": str(bento_model.tag),
                    "mlops.model.accuracy": accuracy,
                }
            )
            training_runs.add(1, {**metric_attributes, "status": "success"})
            print(f"Test accuracy: {accuracy:.3f}")
            print(f"MLflow run: {run.info.run_id}")
            print(f"BentoML model: {bento_model.tag}")
            return str(bento_model.tag)
        except Exception as error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            training_runs.add(1, {**metric_attributes, "status": "error"})
            training_failures.add(1, metric_attributes)
            raise
        finally:
            training_duration.record(time.perf_counter() - started_at, metric_attributes)


@flow(name="iris-training-pipeline", log_prints=True)
def training_pipeline(
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    with tracer.start_as_current_span("training.iris-training-pipeline") as span:
        span.set_attributes(
            {
                "prefect.flow.name": "iris-training-pipeline",
                "mlops.training.n_estimators": n_estimators,
                "mlops.training.test_size": test_size,
                "mlops.training.random_state": random_state,
            }
        )
        try:
            return train_and_register_model(
                n_estimators=n_estimators,
                test_size=test_size,
                random_state=random_state,
            )
        except Exception as error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise


if __name__ == "__main__":
    training_pipeline()
import os
import bentoml
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


@bentoml.service(
    name="iris-classifier",
    tracing={
        "exporter_type": "otlp",
        "sample_rate": 1.0,
        "otlp": {
            "protocol": "http",
            "endpoint": "http://otel-collector:4318/v1/traces",
        },
    },
)
class IrisClassifier:
    def __init__(self) -> None:
        model_tag = os.getenv("BENTOML_MODEL_TAG", "iris_classifier:latest")
        model_info = bentoml.models.get(model_tag)
        self.model_tag = str(model_info.tag)
        metadata = model_info.info.metadata or {}
        self.mlflow_run_id = str(metadata.get("mlflow_run_id", ""))
        self.model = bentoml.sklearn.load_model(self.model_tag)
        self.tracer = trace.get_tracer("mlops.bentoml")

    @bentoml.api
    def predict(self, features: list[list[float]]) -> dict[str, list | str]:
        with self.tracer.start_as_current_span("model.predict") as span:
            span.set_attribute("mlops.model.tag", self.model_tag)
            if self.mlflow_run_id:
                span.set_attribute("mlflow.run_id", self.mlflow_run_id)
            span.set_attribute("mlops.inference.batch_size", len(features))
            try:
                predictions = self.model.predict(features).tolist()
            except Exception as error:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))
                raise

        class_names = ["setosa", "versicolor", "virginica"]
        return {
            "predictions": predictions,
            "class_names": [class_names[index] for index in predictions],
            "model_tag": self.model_tag,
            "mlflow_run_id": self.mlflow_run_id,
        }


svc = IrisClassifier
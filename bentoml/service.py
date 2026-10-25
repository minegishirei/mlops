import os
from functools import cached_property

import bentoml
import mlflow.sklearn


@bentoml.service(name="iris-classifier")
class IrisClassifier:
    def __init__(self) -> None:
        self.model_uri = os.getenv(
            "MLFLOW_MODEL_URI", "models:/iris-classifier@champion"
        )

    @cached_property
    def model(self):
        return mlflow.sklearn.load_model(self.model_uri)

    @bentoml.api
    def predict(self, features: list[list[float]]) -> dict[str, list]:
        predictions = self.model.predict(features).tolist()
        class_names = ["setosa", "versicolor", "virginica"]
        return {
            "predictions": predictions,
            "class_names": [class_names[index] for index in predictions],
        }


svc = IrisClassifier
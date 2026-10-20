import os
import bentoml


@bentoml.service(name="iris-classifier")
class IrisClassifier:
    def __init__(self) -> None:
        model_tag = os.getenv("BENTOML_MODEL_TAG", "iris_classifier:latest")
        self.model = bentoml.sklearn.load_model(model_tag)

    @bentoml.api
    def predict(self, features: list[list[float]]) -> dict[str, list]:
        predictions = self.model.predict(features).tolist()
        class_names = ["setosa", "versicolor", "virginica"]
        return {
            "predictions": predictions,
            "class_names": [class_names[index] for index in predictions],
        }


svc = IrisClassifier
import bentoml


@bentoml.service(name="iris-classifier")
class IrisClassifier:
    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model("iris_classifier:latest")

    @bentoml.api
    def predict(self, features: list[list[float]]) -> dict[str, list]:
        predictions = self.model.predict(features).tolist()
        class_names = ["setosa", "versicolor", "virginica"]
        return {
            "predictions": predictions,
            "class_names": [class_names[index] for index in predictions],
        }


svc = IrisClassifier
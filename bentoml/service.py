import os
import bentoml
import mlflow.sklearn


@bentoml.service(name="iris-classifier")
class IrisClassifier:
    def __init__(self) -> None:
        # 環境変数からMLflowのモデルURIを取得（デフォルトはレジストリのProductionステージ）
        model_uri = os.getenv("MLFLOW_MODEL_URI", None)
        # MLflowモデルレジストリから直接モデルをロード
        self.model = mlflow.sklearn.load_model(model_uri)

    @bentoml.api
    def predict(self, features: list[list[float]]) -> dict[str, list]:
        predictions = self.model.predict(features).tolist()
        class_names = ["setosa", "versicolor", "virginica"]
        return {
            "predictions": predictions,
            "class_names": [class_names[index] for index in predictions],
        }


svc = IrisClassifier
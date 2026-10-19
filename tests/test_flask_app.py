from api.flask_app import create_app


class StubModel:
    def predict(self, features):
        return {
            "predictions": [0 for _ in features],
            "class_names": ["setosa" for _ in features],
        }


def test_homepage_renders_prediction_form():
    client = create_app(predictor=StubModel().predict).test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert "花のサイズ" in response.get_data(as_text=True)


def test_predict_returns_class_name():
    client = create_app(predictor=StubModel().predict).test_client()

    response = client.post("/api/predict", json={"features": [[5.1, 3.5, 1.4, 0.2]]})

    assert response.status_code == 200
    assert response.json == {"predictions": [0], "class_names": ["setosa"]}


def test_predict_rejects_invalid_feature_count():
    client = create_app(predictor=StubModel().predict).test_client()

    response = client.post("/api/predict", json={"features": [[5.1, 3.5]]})

    assert response.status_code == 400
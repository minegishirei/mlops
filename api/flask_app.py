import math
import os
from typing import Any, Callable

import requests
from flask import Flask, jsonify, render_template, request

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class ModelUnavailable(Exception):
    pass


def predict_with_bentoml(features: list[list[float]]) -> dict[str, list]:
    api_url = os.getenv("BENTOML_API_URL", "http://127.0.0.1:3000").rstrip("/")
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"features": features},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        raise ModelUnavailable from error


def create_app(predictor: Callable[[list[list[float]]], dict[str, list]] | None = None) -> Flask:
    app = Flask(__name__)
    app.config["PREDICTOR"] = predictor or predict_with_bentoml

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/healthz")
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok"}), 200

    @app.post("/api/predict")
    def predict() -> Any:
        payload = request.get_json(silent=True)
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list) or not features:
            return jsonify({"error": "features must be a non-empty array of four-value arrays"}), 400

        for row in features:
            if not isinstance(row, list) or len(row) != 4:
                return jsonify({"error": "each feature row must contain four finite numbers"}), 400
            try:
                valid_values = all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in row
                )
            except (OverflowError, ValueError):
                valid_values = False
            if not valid_values:
                return jsonify({"error": "each feature row must contain four finite numbers"}), 400

        try:
            prediction = app.config["PREDICTOR"](features)
        except ModelUnavailable:
            return jsonify({"error": "BentoML is unavailable. Check the model server."}), 503

        return jsonify(prediction)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
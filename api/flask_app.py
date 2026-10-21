import math
import os
import time
from typing import Any, Callable

import requests
from flask import Flask, g, jsonify, render_template, request
from opentelemetry import trace
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.trace import Status, StatusCode

from telemetry import (
    inference_duration,
    inference_errors,
    inference_requests,
    set_active_model_tag,
    tracer,
)

RequestsInstrumentor().instrument()


class ModelUnavailable(Exception):
    pass


def predict_with_bentoml(features: list[list[float]]) -> dict[str, Any]:
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


def create_app(
    predictor: Callable[[list[list[float]]], dict[str, Any]] | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config["PREDICTOR"] = predictor or predict_with_bentoml
    FlaskInstrumentor().instrument_app(app)

    @app.before_request
    def start_request_timer() -> None:
        if request.endpoint == "predict":
            g.inference_started_at = time.perf_counter()

    @app.after_request
    def record_inference_metrics(response: Any) -> Any:
        if request.endpoint != "predict":
            return response

        duration = time.perf_counter() - getattr(g, "inference_started_at", time.perf_counter())
        model_tag = getattr(g, "model_tag", os.getenv("BENTOML_MODEL_TAG", "unknown"))
        attributes = {
            "http.request.method": request.method,
            "http.route": "/api/predict",
            "http.response.status_code": response.status_code,
            "model_tag": model_tag,
        }
        inference_requests.add(1, attributes)
        inference_duration.record(duration, attributes)
        if response.status_code >= 400:
            inference_errors.add(1, attributes)

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("mlops.model.tag", model_tag)
            span.set_attribute("mlops.inference.duration_seconds", duration)
            if response.status_code >= 500:
                span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
        span_context = span.get_span_context()
        if span_context.is_valid:
            response.headers["X-Trace-ID"] = f"{span_context.trace_id:032x}"
        return response

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

        with tracer.start_as_current_span("mlops.inference.request") as span:
            span.set_attribute("mlops.inference.batch_size", len(features))
            try:
                prediction = app.config["PREDICTOR"](features)
            except ModelUnavailable:
                span.set_status(Status(StatusCode.ERROR, "BentoML unavailable"))
                return jsonify({"error": "BentoML is unavailable. Check the model server."}), 503

            model_tag = prediction.get("model_tag")
            mlflow_run_id = prediction.get("mlflow_run_id")
            if isinstance(model_tag, str):
                g.model_tag = model_tag
                set_active_model_tag(model_tag)
                span.set_attribute("mlops.model.tag", model_tag)
            if isinstance(mlflow_run_id, str):
                span.set_attribute("mlflow.run_id", mlflow_run_id)

        return jsonify(prediction)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
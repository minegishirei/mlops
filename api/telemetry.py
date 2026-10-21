import os
import threading

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_active_model_lock = threading.Lock()
_active_model_tag = os.getenv("BENTOML_MODEL_TAG", "unknown")


def configure_telemetry(service_name: str) -> None:
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return

    resource = Resource.create({"service.name": service_name, "service.namespace": "mlops"})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))


def set_active_model_tag(model_tag: str) -> None:
    global _active_model_tag
    with _active_model_lock:
        _active_model_tag = model_tag


def observe_active_model_tag(options: object) -> list[Observation]:
    with _active_model_lock:
        model_tag = _active_model_tag
    return [Observation(1, {"model_tag": model_tag})] if model_tag != "unknown" else []


configure_telemetry("mlops-api")
tracer = trace.get_tracer("mlops.api")
meter = metrics.get_meter("mlops.api")
inference_requests = meter.create_counter("mlops.inference.requests", unit="{request}")
inference_errors = meter.create_counter("mlops.inference.errors", unit="{error}")
inference_duration = meter.create_histogram("mlops.inference.duration", unit="s")
active_model_info = meter.create_observable_gauge(
    "mlops.model.info",
    callbacks=[observe_active_model_tag],
    unit="{model}",
    description="The BentoML model tag currently used by the inference API.",
)
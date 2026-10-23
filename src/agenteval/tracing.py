from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        records: list[str] = []
        with self._lock:
            for span in spans:
                context = span.get_span_context()
                if context is None:
                    continue
                parent = span.parent
                parent_span_id = format(parent.span_id, "016x") if parent is not None else None
                record: dict[str, Any] = {
                    "name": span.name,
                    "trace_id": format(context.trace_id, "032x"),
                    "span_id": format(context.span_id, "016x"),
                    "parent_span_id": parent_span_id,
                    "start_time_ns": span.start_time,
                    "end_time_ns": span.end_time,
                    "status": span.status.status_code.name,
                    "attributes": dict(span.attributes or {}),
                    "events": [
                        {
                            "name": event.name,
                            "timestamp_ns": event.timestamp,
                            "attributes": dict(event.attributes or {}),
                        }
                        for event in span.events
                    ],
                }
                records.append(json.dumps(record, default=str, sort_keys=True))
            if records:
                descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
                try:
                    os.write(descriptor, ("\n".join(records) + "\n").encode())
                finally:
                    os.close(descriptor)
        return SpanExportResult.SUCCESS


def configure_tracing(path: Path) -> trace.Tracer:
    provider = TracerProvider(resource=Resource.create({"service.name": "agenteval"}))
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(path)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("agenteval")


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("agenteval")

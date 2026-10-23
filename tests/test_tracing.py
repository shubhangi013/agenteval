import json
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from agenteval.tracing import JsonlSpanExporter
from agenteval.trajectory import (
    SpanRecord,
    TrajectoryExpectation,
    evaluate_run_trajectory,
    evaluate_trajectory,
)


def test_jsonl_export_preserves_parent_links(tmp_path: Path) -> None:
    path = tmp_path / "spans.jsonl"
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(path)))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("parent"):
        with tracer.start_as_current_span("child"):
            pass
    provider.shutdown()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_name = {record["name"]: record for record in records}
    assert by_name["child"]["trace_id"] == by_name["parent"]["trace_id"]
    assert by_name["child"]["parent_span_id"] == by_name["parent"]["span_id"]


def span(
    name: str,
    span_id: str,
    *,
    tool: str | None = None,
    fingerprint: str = "",
    attempt: int | None = None,
    status: str = "UNSET",
) -> SpanRecord:
    attributes: dict[str, object] = {"run.id": "run-1"}
    if tool is not None:
        attributes.update({"tool.name": tool, "tool.input_hash": fingerprint})
    if attempt is not None:
        attributes["attempt"] = attempt
    return SpanRecord(
        name=name,
        trace_id="trace",
        span_id=span_id,
        start_time_ns=int(span_id),
        status=status,
        attributes=attributes,
    )


def test_trajectory_scores_tools_retries_redundancy_and_efficiency() -> None:
    spans = [
        span("tool.call", "1", tool="search", fingerprint="same"),
        span("reliability.attempt", "2", attempt=1),
        span("tool.call", "3", tool="search", fingerprint="same"),
        span("tool.call", "4", tool="answer", fingerprint="final"),
    ]
    metrics = evaluate_trajectory(
        spans,
        TrajectoryExpectation(
            expected_tools=["search", "answer"],
            allowed_tools=["search", "answer"],
            max_tool_calls=3,
            max_retries=1,
            max_redundant_calls=0,
        ),
    )
    assert metrics.available
    assert metrics.tool_selection_recall == 1.0
    assert metrics.retry_count == 1
    assert metrics.redundant_tool_calls == 1
    assert metrics.navigation_efficiency == 2 / 3
    assert not metrics.passed


def test_trace_reader_reconstructs_full_trace_and_tolerates_corruption(tmp_path: Path) -> None:
    path = tmp_path / "spans.jsonl"
    records = [
        span("planner.decision", "1"),
        span("worker.task", "2"),
        SpanRecord(name="other", trace_id="other", span_id="3"),
    ]
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n{corrupt\n",
        encoding="utf-8",
    )
    metrics = evaluate_run_trajectory(path, "run-1")
    assert metrics.available
    assert metrics.span_count == 2
    assert metrics.malformed_span_count == 1

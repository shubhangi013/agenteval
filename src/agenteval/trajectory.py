from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class SpanEvent(BaseModel):
    name: str
    timestamp_ns: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class SpanRecord(BaseModel):
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    start_time_ns: int | None = None
    end_time_ns: int | None = None
    status: str = "UNSET"
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)


class TrajectoryExpectation(BaseModel):
    expected_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    matching_mode: Literal["exact", "in_order", "any_order"] = "in_order"
    max_tool_calls: int | None = Field(default=None, ge=0)
    max_retries: int | None = Field(default=None, ge=0)
    max_redundant_calls: int | None = Field(default=None, ge=0)


class TrajectoryMetrics(BaseModel):
    available: bool
    span_count: int = 0
    malformed_span_count: int = 0
    tool_names: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    successful_tool_calls: int = 0
    tool_success_rate: float = 0.0
    retry_count: int = 0
    redundant_tool_calls: int = 0
    tool_selection_precision: float | None = None
    tool_selection_recall: float | None = None
    tool_selection_f1: float | None = None
    navigation_efficiency: float = 0.0
    recovery_success: bool | None = None
    passed: bool = True
    violations: list[str] = Field(default_factory=list)


def _trace_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.glob("*.jsonl"))


def load_spans(path: Path) -> tuple[list[SpanRecord], int]:
    spans: list[SpanRecord] = []
    malformed = 0
    for trace_file in _trace_files(path):
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                spans.append(SpanRecord.model_validate_json(line))
            except (ValueError, json.JSONDecodeError):
                malformed += 1
    return spans, malformed


def reconstruct_trajectory(path: Path, run_id: str) -> tuple[list[SpanRecord], int]:
    spans, malformed = load_spans(path)
    trace_ids = {
        span.trace_id for span in spans if str(span.attributes.get("run.id", "")) == run_id
    }
    selected = [span for span in spans if span.trace_id in trace_ids]
    selected.sort(key=lambda span: span.start_time_ns or 0)
    return selected, malformed


def _sequence_matches(actual: list[str], expected: list[str], mode: str) -> bool:
    if mode == "exact":
        return actual == expected
    if mode == "any_order":
        return Counter(actual) == Counter(expected)
    iterator = iter(actual)
    return all(any(item == wanted for item in iterator) for wanted in expected)


def evaluate_trajectory(
    spans: list[SpanRecord],
    expectation: TrajectoryExpectation | None = None,
    *,
    malformed_span_count: int = 0,
) -> TrajectoryMetrics:
    if not spans:
        return TrajectoryMetrics(available=False, malformed_span_count=malformed_span_count)

    expected = expectation or TrajectoryExpectation()
    tool_spans = [span for span in spans if span.name == "tool.call"]
    tool_names = [str(span.attributes.get("tool.name", "unknown")) for span in tool_spans]
    successful = sum(span.status != "ERROR" for span in tool_spans)
    retry_count = sum(
        1
        for span in spans
        if span.name == "reliability.attempt" and int(span.attributes.get("attempt", 0)) > 0
    )
    fingerprints = [
        (
            str(span.attributes.get("tool.name", "unknown")),
            str(span.attributes.get("tool.input_hash", "")),
        )
        for span in tool_spans
        if span.attributes.get("tool.input_hash")
    ]
    redundant = sum(max(0, count - 1) for count in Counter(fingerprints).values())
    violations: list[str] = []

    if expected.expected_tools and not _sequence_matches(
        tool_names, expected.expected_tools, expected.matching_mode
    ):
        violations.append(
            f"tool sequence {tool_names!r} does not satisfy {expected.matching_mode} "
            f"expectation {expected.expected_tools!r}"
        )
    if expected.allowed_tools:
        unexpected = sorted(set(tool_names) - set(expected.allowed_tools))
        if unexpected:
            violations.append(f"unexpected tools called: {unexpected!r}")
    forbidden = sorted(set(tool_names) & set(expected.forbidden_tools))
    if forbidden:
        violations.append(f"forbidden tools called: {forbidden!r}")
    if expected.max_tool_calls is not None and len(tool_spans) > expected.max_tool_calls:
        violations.append(
            f"tool call count {len(tool_spans)} exceeds maximum {expected.max_tool_calls}"
        )
    if expected.max_retries is not None and retry_count > expected.max_retries:
        violations.append(f"retry count {retry_count} exceeds maximum {expected.max_retries}")
    if expected.max_redundant_calls is not None and redundant > expected.max_redundant_calls:
        violations.append(
            f"redundant tool calls {redundant} exceed maximum {expected.max_redundant_calls}"
        )
    failed_calls = len(tool_spans) - successful
    if failed_calls:
        violations.append(f"{failed_calls} tool call(s) failed")

    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    if expected.expected_tools:
        intersection = sum(
            (Counter(tool_names) & Counter(expected.expected_tools)).values()
        )
        precision = intersection / len(tool_names) if tool_names else 0.0
        recall = intersection / len(expected.expected_tools)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    useful_calls = max(0, successful - redundant)
    navigation_efficiency = useful_calls / len(tool_spans) if tool_spans else 1.0
    recovery_success = None if retry_count == 0 else successful > 0
    return TrajectoryMetrics(
        available=True,
        span_count=len(spans),
        malformed_span_count=malformed_span_count,
        tool_names=tool_names,
        tool_call_count=len(tool_spans),
        successful_tool_calls=successful,
        tool_success_rate=successful / len(tool_spans) if tool_spans else 1.0,
        retry_count=retry_count,
        redundant_tool_calls=redundant,
        tool_selection_precision=precision,
        tool_selection_recall=recall,
        tool_selection_f1=f1,
        navigation_efficiency=navigation_efficiency,
        recovery_success=recovery_success,
        passed=not violations,
        violations=violations,
    )


def evaluate_run_trajectory(
    path: Path, run_id: str, expectation: TrajectoryExpectation | None = None
) -> TrajectoryMetrics:
    spans, malformed = reconstruct_trajectory(path, run_id)
    return evaluate_trajectory(spans, expectation, malformed_span_count=malformed)
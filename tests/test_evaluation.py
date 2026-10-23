from datetime import UTC, datetime

from agenteval.evaluation import (
    EvalResult,
    EvalRun,
    GoldenTask,
    combine_signals,
    heuristic_score,
    regression_deltas,
)
from agenteval.reporting import summarize_reliability, wilson_interval


def task(**changes: object) -> GoldenTask:
    values: dict[str, object] = {
        "id": "task",
        "category": "factual_qa",
        "input": {"goal": "code: result = 'Paris'"},
        "rubric": "capital of France",
        "reference_answers": ["Paris"],
        "regex": "^Paris$",
        "cosine_threshold": 1.0,
    }
    values.update(changes)
    return GoldenTask.model_validate(values)


def result(passed: bool, probability: float) -> EvalResult:
    return EvalResult(
        task_id="task",
        category="factual_qa",
        output="Paris",
        llm_score=5 if passed else 0,
        heuristic_score=1 if passed else 0,
        final_probability=probability,
        passed=passed,
        disagreement=False,
        latency_ms=1,
        cost_usd=0,
        run_id="run",
    )


def result_for(
    task_id: str, category: str, passed: bool, probability: float, trial: int
) -> EvalResult:
    item = result(passed, probability)
    item.task_id = task_id
    item.category = category
    item.trial = trial
    return item


def run(item: EvalResult) -> EvalRun:
    return EvalRun(
        created_at=datetime.now(UTC),
        concurrency=1,
        results=[item],
        total_cost_usd=0,
        p50_latency_ms=1,
        p95_latency_ms=1,
        disagreement_rate=0,
    )


def test_heuristic_uses_regex_similarity_and_refusal() -> None:
    assert heuristic_score(task(), "Paris") == 1.0
    assert heuristic_score(task(), "I cannot answer") < 0.5


def test_sigmoid_combines_both_signals() -> None:
    assert combine_signals(5, 1) > 0.9
    assert combine_signals(0, 0) < 0.01


def test_regression_detects_passing_to_failing_transition() -> None:
    lines, regressed = regression_deltas(run(result(True, 0.9)), run(result(False, 0.1)))
    assert regressed
    assert "-1" in lines[1]


def test_reliability_reports_pass_at_k_pass_to_k_variance_and_interval() -> None:
    summary = summarize_reliability(
        [
            result_for("task", "qa", True, 0.9, 1),
            result_for("task", "qa", False, 0.3, 2),
            result_for("task", "qa", True, 0.6, 3),
        ]
    )[0]
    assert summary.attempts == 3
    assert summary.pass_at_k
    assert not summary.pass_to_k
    assert summary.pass_rate == 2 / 3
    assert summary.probability_variance > 0
    assert summary.confidence_low < summary.pass_rate < summary.confidence_high


def test_wilson_interval_handles_no_attempts() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_category_pass_rate_drop_is_a_regression() -> None:
    baseline = run(result_for("one", "accuracy", True, 0.9, 1))
    candidate = run(result_for("one", "accuracy", False, 0.8, 1))
    lines, regressed = regression_deltas(
        baseline, candidate, max_category_pass_rate_drop=0.5
    )
    assert regressed
    assert any("category:accuracy" in line for line in lines)


def test_legacy_eval_run_loads_with_richer_defaults() -> None:
    loaded = EvalRun.model_validate(run(result(True, 0.9)).model_dump(exclude={"reliability"}))
    assert loaded.repetitions == 1
    assert loaded.reliability == []

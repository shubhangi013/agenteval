from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from agenteval.llm import LLMClient, parse_json_response
from agenteval.reporting import (
    CategorySummary,
    ReliabilitySummary,
    summarize_categories,
    summarize_reliability,
)
from agenteval.trajectory import (
    TrajectoryExpectation,
    TrajectoryMetrics,
    evaluate_run_trajectory,
)


class GoldenTask(BaseModel):
    id: str
    category: str
    input: dict[str, Any]
    rubric: str
    reference_answers: list[str]
    regex: str | None = None
    cosine_threshold: float = 0.6
    unit_tests: str | None = None
    min_llm_score: float = Field(default=3.0, ge=0, le=5)
    min_heuristic_score: float = Field(default=0.6, ge=0, le=1)
    trajectory: TrajectoryExpectation | None = None


class EvalResult(BaseModel):
    task_id: str
    category: str
    output: str
    llm_score: float
    heuristic_score: float
    final_probability: float
    passed: bool
    disagreement: bool
    latency_ms: float
    cost_usd: float
    run_id: str
    error: str | None = None
    trial: int = 1
    judge_reason: str = ""
    agent_latency_ms: float = 0.0
    judge_latency_ms: float = 0.0
    agent_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    trajectory: TrajectoryMetrics | None = None


class EvalRun(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    concurrency: int
    results: list[EvalResult]
    total_cost_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    disagreement_rate: float
    repetitions: int = 1
    pass_rate: float = 0.0
    pass_at_k: float = 0.0
    pass_to_k: float = 0.0
    reliability: list[ReliabilitySummary] = Field(default_factory=list)
    categories: list[CategorySummary] = Field(default_factory=list)


def load_goldens(path: Path) -> list[GoldenTask]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenTask.model_validate(item) for item in data]


def _tokens(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def cosine_similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    common = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _passes_unit_tests(code: str, tests: str) -> bool:
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "candidate_test.py"
        script.write_text(code + "\n" + tests, encoding="utf-8")
        completed = subprocess.run(
            ["python", str(script)], capture_output=True, text=True, timeout=3, check=False
        )
        return completed.returncode == 0


def heuristic_score(task: GoldenTask, output: str) -> float:
    checks: list[bool] = []
    if task.regex:
        checks.append(re.search(task.regex, output, re.I | re.S) is not None)
    similarities = [cosine_similarity(output, reference) for reference in task.reference_answers]
    checks.append(max(similarities, default=0.0) >= task.cosine_threshold)
    refusal = re.search(r"\b(i (?:cannot|can't|won't)|unable to|as an ai)\b", output, re.I)
    checks.append(refusal is None)
    if task.unit_tests:
        checks.append(_passes_unit_tests(output, task.unit_tests))
    return sum(checks) / len(checks)


def combine_signals(llm_score: float, heuristic: float) -> float:
    weighted = 0.55 * (llm_score / 5.0) + 0.45 * heuristic
    return 1.0 / (1.0 + math.exp(-8.0 * (weighted - 0.65)))


class Evaluator:
    def __init__(
        self,
        base_url: str,
        judge: LLMClient,
        disagreement_dir: Path = Path("disagreements"),
        trace_path: Path = Path("traces"),
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.judge = judge
        self.disagreement_dir = disagreement_dir
        self.trace_path = trace_path

    async def evaluate(
        self, tasks: list[GoldenTask], concurrency: int, repetitions: int = 1
    ) -> EvalRun:
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        semaphore = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=120) as client:
            async def bounded(task: GoldenTask, trial: int) -> EvalResult:
                async with semaphore:
                    return await self._evaluate_one(client, task, trial)

            results = await asyncio.gather(
                *(
                    bounded(task, trial)
                    for task in tasks
                    for trial in range(1, repetitions + 1)
                )
            )
        latencies = sorted(result.latency_ms for result in results)
        reliability = summarize_reliability(results)
        return EvalRun(
            concurrency=concurrency,
            results=results,
            total_cost_usd=sum(result.cost_usd for result in results),
            p50_latency_ms=_percentile(latencies, 50),
            p95_latency_ms=_percentile(latencies, 95),
            disagreement_rate=sum(result.disagreement for result in results) / len(results),
            repetitions=repetitions,
            pass_rate=sum(result.passed for result in results) / len(results),
            pass_at_k=sum(item.pass_at_k for item in reliability) / len(reliability),
            pass_to_k=sum(item.pass_to_k for item in reliability) / len(reliability),
            reliability=reliability,
            categories=summarize_categories(results),
        )

    async def _evaluate_one(
        self, client: httpx.AsyncClient, task: GoldenTask, trial: int = 1
    ) -> EvalResult:
        started = time.monotonic()
        agent_started = time.monotonic()
        run_id = ""
        error: str | None = None
        output = ""
        cost = 0.0
        try:
            response = await client.post(f"{self.base_url}/v1/runs", json=task.input)
            response.raise_for_status()
            run_id = str(response.json()["run"]["id"])
            while True:
                status = await client.get(f"{self.base_url}/v1/runs/{run_id}")
                status.raise_for_status()
                payload = status.json()
                state = payload["run"]["status"]
                if state in {"completed", "failed", "cancelled", "budget_exceeded"}:
                    cost = float(payload["run"].get("cost_usd", 0.0))
                    if payload.get("results"):
                        result = payload["results"][-1]
                        error = result.get("error")
                        output = _extract_output(result.get("output", {}))
                    break
                await asyncio.sleep(0.02)
        except Exception as exc:
            error = str(exc)
        agent_latency_ms = (time.monotonic() - agent_started) * 1000
        trajectory = (
            evaluate_run_trajectory(self.trace_path, run_id, task.trajectory)
            if run_id
            else None
        )
        prompt = (
            "JUDGE\nScore the candidate from 0 to 5 against the rubric. "
            "Return JSON with score and reason.\n"
            f"Rubric: {task.rubric}\nReferences: {task.reference_answers}\nCandidate: {output}"
        )
        judge_started = time.monotonic()
        judge_response = await self.judge.complete(prompt, json_mode=True)
        judge_latency_ms = (time.monotonic() - judge_started) * 1000
        judged = parse_json_response(judge_response)
        agent_cost = cost
        judge_cost = judge_response.cost_usd
        cost += judge_cost
        llm_score = max(0.0, min(5.0, float(judged["score"])))
        heuristic = heuristic_score(task, output)
        probability = combine_signals(llm_score, heuristic)
        llm_passed = llm_score >= task.min_llm_score
        heuristic_passed = heuristic >= task.min_heuristic_score
        trajectory_passed = (
            task.trajectory is None
            or trajectory is None
            or not trajectory.available
            or trajectory.passed
        )
        disagreement = llm_passed != heuristic_passed
        result = EvalResult(
            task_id=task.id,
            category=task.category,
            output=output,
            llm_score=llm_score,
            heuristic_score=heuristic,
            final_probability=probability,
            passed=llm_passed and heuristic_passed and trajectory_passed and error is None,
            disagreement=disagreement,
            latency_ms=(time.monotonic() - started) * 1000,
            cost_usd=cost,
            run_id=run_id,
            error=error,
            trial=trial,
            judge_reason=str(judged.get("reason", "")),
            agent_latency_ms=agent_latency_ms,
            judge_latency_ms=judge_latency_ms,
            agent_cost_usd=agent_cost,
            judge_cost_usd=judge_cost,
            trajectory=trajectory,
        )
        if disagreement:
            self._write_disagreement(task, result)
        return result

    def _write_disagreement(self, task: GoldenTask, result: EvalResult) -> None:
        target = self.disagreement_dir / task.id
        target.mkdir(parents=True, exist_ok=True)
        (target / "scores.json").write_text(
            json.dumps(
                {"task": task.model_dump(mode="json"), "result": result.model_dump(mode="json")},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        trace_files = (
            [self.trace_path]
            if self.trace_path.is_file()
            else sorted(self.trace_path.glob("*.jsonl"))
        )
        if trace_files:
            trace_lines: list[str] = []
            for trace_file in trace_files:
                for line in trace_file.read_text(encoding="utf-8").splitlines():
                    if result.run_id in line:
                        trace_lines.append(line)
            (target / "trace.jsonl").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")


def _extract_output(output: dict[str, Any]) -> str:
    if "result" in output:
        return str(output["result"])
    if "value" in output:
        value = output["value"]
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    if "text" in output:
        return str(output["text"])
    return json.dumps(output, sort_keys=True)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    index = math.ceil((percentile / 100) * len(values)) - 1
    return values[max(0, index)]


def save_eval_run(run: EvalRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def regression_deltas(
    baseline: EvalRun,
    candidate: EvalRun,
    *,
    max_category_pass_rate_drop: float = 0.0,
    max_cost_increase_ratio: float | None = None,
    max_p95_latency_increase_ratio: float | None = None,
) -> tuple[list[str], bool]:
    previous = {result.task_id: result for result in baseline.results}
    lines = ["task_id\tpass_delta\tscore_delta\tlatency_delta_ms"]
    regressed = False
    for current in candidate.results:
        old = previous.get(current.task_id)
        if old is None:
            continue
        pass_delta = int(current.passed) - int(old.passed)
        score_delta = current.final_probability - old.final_probability
        latency_delta = current.latency_ms - old.latency_ms
        lines.append(
            f"{current.task_id}\t{pass_delta:+d}\t{score_delta:+.3f}\t{latency_delta:+.1f}"
        )
        if old.passed and not current.passed:
            regressed = True
    old_categories = {item.category: item for item in summarize_categories(baseline.results)}
    new_categories = {item.category: item for item in summarize_categories(candidate.results)}
    for category, current_category in sorted(new_categories.items()):
        old_category = old_categories.get(category)
        if old_category is None:
            continue
        delta = current_category.pass_rate - old_category.pass_rate
        lines.append(f"category:{category}\t{delta:+.3f}\tpass_rate\t-")
        if delta < -max_category_pass_rate_drop:
            regressed = True
    if max_cost_increase_ratio is not None and baseline.total_cost_usd > 0:
        increase = candidate.total_cost_usd / baseline.total_cost_usd - 1
        if increase > max_cost_increase_ratio:
            lines.append(f"run:cost\t-\t{increase:+.1%}\t-")
            regressed = True
    if max_p95_latency_increase_ratio is not None and baseline.p95_latency_ms > 0:
        increase = candidate.p95_latency_ms / baseline.p95_latency_ms - 1
        if increase > max_p95_latency_increase_ratio:
            lines.append(f"run:p95_latency\t-\t-\t{increase:+.1%}")
            regressed = True
    return lines, regressed

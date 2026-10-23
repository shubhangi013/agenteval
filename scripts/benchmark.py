from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from agenteval.config import Settings
from agenteval.evaluation import EvalRun, Evaluator, load_goldens, save_eval_run
from agenteval.llm import create_llm_client

BASE_URL = os.getenv("AGENTEVAL_BASE_URL", "http://localhost:8000")
LEVELS = [int(value) for value in os.getenv("BENCH_LEVELS", "1,8,16,32").split(",")]
TASK_LIMIT = int(os.getenv("BENCH_TASK_LIMIT", "0"))
COMPOSE_COMMAND = ["docker-compose"] if shutil.which("docker-compose") else ["docker", "compose"]


def compose(*args: str) -> None:
    subprocess.run([*COMPOSE_COMMAND, *args], check=True)


async def wait_healthy(timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{BASE_URL}/health", timeout=2)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError("AgentEval API did not become healthy")


async def wait_run(client: httpx.AsyncClient, run_id: str, timeout_s: float = 45) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = await client.get(f"{BASE_URL}/v1/runs/{run_id}")
        payload = response.json()
        if payload["run"]["status"] in {"completed", "failed", "cancelled", "budget_exceeded"}:
            return payload["run"]["status"] == "completed"
        await asyncio.sleep(0.1)
    return False


async def inject_failure(mode: str) -> bool:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{BASE_URL}/v1/runs",
            json={
                "goal": "calculate: 6 * 7",
                "max_latency_ms": 45_000,
                "metadata": {"failure_mode": mode},
            },
        )
        response.raise_for_status()
        run_id = str(response.json()["run"]["id"])
        if mode == "worker_crash":
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                health = (await client.get(f"{BASE_URL}/health")).json()
                if int(health["processing"]) > 0:
                    break
                await asyncio.sleep(0.1)
            compose("kill", "worker")
            compose("up", "-d", "--scale", "worker=1", "worker")
        return await wait_run(client, run_id)


def report(runs: list[EvalRun], recovery: dict[str, bool]) -> str:
    latest = runs[-1]
    rows = [
        "| Workers | Passed | Failed | p50 latency (ms) | p95 latency (ms) | "
        "Disagreement | Cost (USD) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        passed = sum(result.passed for result in run.results)
        rows.append(
            f"| {run.concurrency} | {passed} | {len(run.results) - passed} | "
            f"{run.p50_latency_ms:.1f} | {run.p95_latency_ms:.1f} | "
            f"{run.disagreement_rate:.1%} | {run.total_cost_usd:.6f} |"
        )
    recovery_rows = ["| Failure mode | Recovered |", "|---|---:|"]
    recovery_rows.extend(f"| {name} | {'yes' if ok else 'no'} |" for name, ok in recovery.items())
    recovery_rate = sum(recovery.values()) / len(recovery)
    return "\n".join(
        [
            "# AgentEval Benchmark Report",
            "",
            f"Generated from {len(latest.results)} golden tasks using the "
            "configured judge provider.",
            "",
            "## Concurrency",
            "",
            *rows,
            "",
            "## Recovery",
            "",
            *recovery_rows,
            "",
            f"**Recovery success rate:** {recovery_rate:.1%}",
            f"**Judge disagreement rate:** {latest.disagreement_rate:.1%}",
            f"**Total measured cost:** ${sum(run.total_cost_usd for run in runs):.6f}",
            "",
            "Raw per-task measurements are stored under `runs/`; OTel spans are under `traces/`.",
        ]
    )


async def main() -> None:
    await wait_healthy()
    settings = Settings()
    tasks = load_goldens(Path("goldens/tasks.json"))
    if TASK_LIMIT:
        tasks = tasks[:TASK_LIMIT]
    evaluator = Evaluator(BASE_URL, create_llm_client(settings.judge_provider, settings))
    runs: list[EvalRun] = []
    for level in LEVELS:
        compose("up", "-d", "--scale", f"worker={level}", "worker")
        await wait_healthy()
        result = await evaluator.evaluate(tasks, level)
        save_eval_run(result, Path(f"runs/concurrency-{level}.json"))
        runs.append(result)
    recovery = {
        mode: await inject_failure(mode)
        for mode in ("tool_timeout", "llm_429", "worker_crash")
    }
    report_text = report(runs, recovery)
    await asyncio.to_thread(Path("bench_report.md").write_text, report_text, encoding="utf-8")
    print(report_text)
    if not all(recovery.values()):
        raise SystemExit("one or more failure recovery checks failed")


if __name__ == "__main__":
    asyncio.run(main())

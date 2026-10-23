from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from agenteval.bootstrap import create_store
from agenteval.config import Settings
from agenteval.discovery import (
    discover_external_suite,
    generate_adversarial_suite,
    load_json_evidence,
    load_openapi,
)
from agenteval.distributed import DistributedExternalAgentEvaluator
from agenteval.evaluation import EvalRun, regression_deltas
from agenteval.external import (
    ExternalAgentEvaluator,
    ExternalEvalRun,
    ExternalSuite,
    HttpAgentTarget,
    load_external_suite,
)
from agenteval.llm import create_llm_client
from agenteval.trajectory import evaluate_run_trajectory

app = typer.Typer(help="AgentEval runtime and evaluation CLI")
console = Console()


@app.callback()
def callback() -> None:
    """Run and compare AgentEval evaluations."""


@app.command("eval")
def compare(
    baseline: Path = typer.Option(..., exists=True, readable=True),
    candidate: Path = typer.Option(..., exists=True, readable=True),
    max_category_pass_rate_drop: float = typer.Option(0.0, min=0.0, max=1.0),
    max_cost_increase_ratio: float | None = typer.Option(None, min=0.0),
    max_p95_latency_increase_ratio: float | None = typer.Option(None, min=0.0),
) -> None:
    """Compare two evaluation runs and fail when a passing task regresses."""
    old = EvalRun.model_validate_json(baseline.read_text(encoding="utf-8"))
    new = EvalRun.model_validate_json(candidate.read_text(encoding="utf-8"))
    lines, regressed = regression_deltas(
        old,
        new,
        max_category_pass_rate_drop=max_category_pass_rate_drop,
        max_cost_increase_ratio=max_cost_increase_ratio,
        max_p95_latency_increase_ratio=max_p95_latency_increase_ratio,
    )
    console.print("\n".join(lines))
    if regressed:
        raise typer.Exit(code=1)


@app.command("discover-agent")
def discover_agent(
    target_url: str = typer.Option(..., help="Base URL of the agent deployment."),
    openapi: str | None = typer.Option(
        None, help="OpenAPI JSON URL or local path; defaults to TARGET_URL/openapi.json."
    ),
    notes: Path | None = typer.Option(None, exists=True, readable=True),
    sample: list[Path] | None = typer.Option(
        None, exists=True, readable=True, help="Repeat for captured request/response JSON files."
    ),
    output: Path = typer.Option(Path("goldens/discovered-agent.draft.json")),
) -> None:
    """Use a model to infer and draft an HTTP agent evaluation suite."""
    settings = Settings()
    if settings.llm_provider == "mock":
        raise typer.BadParameter(
            "discovery requires AGENTEVAL_LLM_PROVIDER=openai, azure_openai, anthropic, "
            "or ollama"
        )
    discovery_notes = notes.read_text(encoding="utf-8") if notes else ""
    discovery_samples = load_json_evidence(sample or [])

    async def run() -> ExternalSuite:
        openapi_source = openapi or f"{target_url.rstrip('/')}/openapi.json"
        return await discover_external_suite(
            create_llm_client(settings.llm_provider, settings),
            base_url=target_url,
            openapi=await load_openapi(openapi_source),
            notes=discovery_notes,
            samples=discovery_samples,
        )

    suite = asyncio.run(run())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(suite.model_dump_json(indent=2), encoding="utf-8")
    console.print(
        f"draft={output} suite={suite.name} endpoint={suite.target.endpoint} "
        f"cases={len(suite.cases)}"
    )
    console.print("Review the draft checks and ground truth before evaluating the agent.")


@app.command("generate-adversarial")
def generate_adversarial(
    suite: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(Path("goldens/adversarial-agent.draft.json")),
    case_count: int = typer.Option(8, min=1, max=50),
) -> None:
    """Generate a review-only adversarial safety suite from a reviewed suite."""
    settings = Settings()
    if settings.llm_provider == "mock":
        raise typer.BadParameter(
            "adversarial generation requires AGENTEVAL_LLM_PROVIDER=openai, azure_openai, "
            "anthropic, or ollama"
        )
    source = load_external_suite(suite)
    draft = asyncio.run(
        generate_adversarial_suite(
            create_llm_client(settings.llm_provider, settings),
            source,
            case_count=case_count,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"draft={output} source={source.name} cases={len(draft.cases)}")
    console.print("Review all threat assumptions and safety policies before evaluation.")


@app.command("evaluate-agent")
def evaluate_agent(
    suite: Path = typer.Option(..., exists=True, readable=True),
    target_url: str | None = typer.Option(None),
    output: Path = typer.Option(Path("runs/external-agent-latest.json")),
    concurrency: int = typer.Option(1, min=1, max=32),
    repetitions: int = typer.Option(1, min=1, max=100),
    local: bool = typer.Option(False, help="Bypass Redis DAG workers for diagnostics."),
    timeout_s: float = typer.Option(600.0, min=1.0),
) -> None:
    """Evaluate an HTTP agent through a durable Redis DAG."""
    settings = Settings()
    suite_definition = load_external_suite(suite)
    if suite_definition.draft:
        raise typer.BadParameter(
            "refusing to evaluate an unreviewed draft; review it and set draft=false first"
        )
    resolved_url = target_url or suite_definition.target.base_url
    suite_definition.target.base_url = resolved_url.rstrip("/")

    async def run_evaluation() -> ExternalEvalRun:
        if local:
            evaluator = ExternalAgentEvaluator(
                HttpAgentTarget(suite_definition.target),
                create_llm_client(settings.judge_provider, settings),
                target_name=resolved_url,
            )
            return await evaluator.evaluate(suite_definition, concurrency, repetitions)
        store = create_store(settings)
        try:
            return await DistributedExternalAgentEvaluator(store, settings).evaluate(
                suite_definition,
                repetitions=repetitions,
                timeout_s=timeout_s,
            )
        finally:
            await store.redis.aclose()

    result = asyncio.run(run_evaluation())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    safety_summary = (
        f" safety={result.safety_pass_rate:.1%}"
        if result.safety_pass_rate is not None
        else ""
    )
    console.print(
        f"tasks={len(result.results)} pass_rate={result.pass_rate:.1%} "
        f"pass@{repetitions}={result.pass_at_k:.1%} pass^{repetitions}={result.pass_to_k:.1%} "
        f"disagreement={result.disagreement_rate:.1%} "
        f"p50={result.p50_latency_ms:.1f}ms p95={result.p95_latency_ms:.1f}ms"
        f"{safety_summary} mode={result.execution_mode} nodes={result.dag_node_count}"
    )
    for item in result.results:
        failed = ", ".join(check.name for check in item.checks if not check.passed)
        console.print(
            f"{'PASS' if item.passed else 'FAIL'} {item.task_id} "
            f"heuristic={item.heuristic_score:.2f} llm={item.llm_score:.1f} "
            f"failed=[{failed}]",
            markup=False,
        )
    if any(not item.passed for item in result.results):
        raise typer.Exit(code=1)


@app.command("evaluate-trace")
def evaluate_trace(
    trace_path: Path = typer.Option(..., exists=True, readable=True),
    run_id: str = typer.Option(...),
) -> None:
    """Reconstruct and report deterministic trajectory metrics for one run."""
    result = evaluate_run_trajectory(trace_path, run_id)
    console.print(result.model_dump_json(indent=2))
    if not result.available or not result.passed:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

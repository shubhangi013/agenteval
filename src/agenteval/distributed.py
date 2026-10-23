from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from agenteval.config import Settings
from agenteval.external import (
    ExternalEvalResult,
    ExternalEvalRun,
    ExternalSuite,
    build_external_eval_run,
    expand_external_cases,
)
from agenteval.models import Budget, RunState, RunStatus, TaskSpec
from agenteval.store import RedisStore
from agenteval.tools import ExternalEvaluationInput


class DistributedExternalAgentEvaluator:
    def __init__(self, store: RedisStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    async def evaluate(
        self,
        suite: ExternalSuite,
        *,
        repetitions: int = 1,
        timeout_s: float = 600.0,
    ) -> ExternalEvalRun:
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        run = RunState(
            goal=f"evaluate external suite: {suite.name}",
            budget=Budget(
                max_cost_usd=self.settings.default_max_cost_usd,
                max_latency_ms=int(timeout_s * 1000),
            ),
            metadata={"kind": "external_evaluation", "suite": suite.name},
        )
        await self.store.create_run(run)

        stateful = any(
            case.state_preconditions
            or case.state_postconditions
            or "stateful" in case.threat_tags
            for case in suite.cases
        )
        tasks: list[TaskSpec] = []
        previous_task_id: str | None = None
        for case, perturbation_id in expand_external_cases(suite):
            for trial in range(1, repetitions + 1):
                node_input = ExternalEvaluationInput(
                    target=suite.target,
                    case=case,
                    default_checks=suite.default_checks,
                    min_llm_score=suite.min_llm_score,
                    trial=trial,
                    perturbation_id=perturbation_id,
                )
                task = TaskSpec(
                    run_id=run.id,
                    name=f"evaluate:{case.id}:trial:{trial}",
                    description=f"Evaluate {case.id}, trial {trial}",
                    tool="external_evaluate",
                    tool_input=node_input.model_dump(mode="json"),
                    dependencies=[previous_task_id] if stateful and previous_task_id else [],
                    acceptance_criteria="Evaluation node returns a validated result artifact.",
                    max_attempts=self.settings.max_retries + 1,
                ).with_id()
                tasks.append(task)
                if stateful:
                    previous_task_id = task.id

        barrier = TaskSpec(
            run_id=run.id,
            name="evaluation:barrier",
            description="Complete after every evaluation node is durable.",
            tool="evaluation_barrier",
            tool_input={"expected_nodes": len(tasks)},
            dependencies=[task.id for task in tasks],
            acceptance_criteria="All evaluation nodes completed.",
        ).with_id()
        run.task_ids = [task.id for task in tasks] + [barrier.id]
        run.status = RunStatus.RUNNING
        await self.store.save_run(run)
        for task in [*tasks, barrier]:
            await self.store.enqueue(task)

        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            current = await self.store.refresh_run_status(run.id)
            if current is None:
                raise RuntimeError(f"evaluation run {run.id} disappeared")
            if current.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                break
            await asyncio.sleep(0.1)
        else:
            await self.store.cancel_run(run.id, "distributed evaluation timed out")
            raise TimeoutError(f"distributed evaluation {run.id} timed out after {timeout_s}s")

        current = await self.store.get_run(run.id)
        if current is None or current.status != RunStatus.COMPLETED:
            status = current.status if current else "missing"
            raise RuntimeError(f"distributed evaluation {run.id} ended with status {status}")

        results: list[ExternalEvalResult] = []
        for task in tasks:
            task_result = await self.store.get_result(task.id)
            if task_result is None:
                raise RuntimeError(f"evaluation node {task.id} has no result")
            results.append(
                ExternalEvalResult.model_validate(task_result.output.get("result"))
            )

        artifact = build_external_eval_run(
            target=suite.target.base_url,
            suite=suite.name,
            results=results,
            repetitions=repetitions,
        )
        artifact.execution_mode = "redis_dag"
        artifact.execution_run_id = run.id
        artifact.dag_node_count = len(tasks) + 1
        artifact.completed_at = datetime.now(UTC)
        return artifact
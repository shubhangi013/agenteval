from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from opentelemetry import trace

from agenteval.agents import PlannerAgent
from agenteval.config import Settings
from agenteval.models import Budget, GoalRequest, RunResponse, RunState, RunStatus
from agenteval.store import RedisStore


class AgentRuntime:
    def __init__(self, store: RedisStore, planner: PlannerAgent, settings: Settings) -> None:
        self.store = store
        self.planner = planner
        self.settings = settings

    async def submit(self, request: GoalRequest) -> RunResponse:
        run = RunState(
            goal=request.goal,
            metadata=request.metadata,
            budget=Budget(
                max_cost_usd=request.max_cost_usd,
                max_latency_ms=request.max_latency_ms,
            ),
        )
        with trace.get_tracer("agenteval").start_as_current_span("run.submit") as span:
            span.set_attribute("run.id", run.id)
            span.set_attribute("budget.max_cost_usd", run.budget.max_cost_usd)
            span.set_attribute("budget.max_latency_ms", run.budget.max_latency_ms)
            await self.store.create_run(run)
            plan = await self.planner.plan(run)
            run.task_ids = [task.id for task in plan.tasks]
            run.status = RunStatus.RUNNING
            await self.store.save_run(run)
            for task in plan.tasks:
                await self.store.enqueue(task)
        return RunResponse(run=run, trace_path=str(self.trace_path(run.id)))

    async def status(self, run_id: str) -> RunResponse | None:
        run = await self.store.refresh_run_status(run_id)
        if run is None:
            return None
        elapsed_ms = (datetime.now(UTC) - run.started_at).total_seconds() * 1000
        if run.status == RunStatus.RUNNING and elapsed_ms > run.budget.max_latency_ms:
            await self.store.cancel_run(run.id, "latency budget exceeded")
            run.status = RunStatus.BUDGET_EXCEEDED
            run.error = "latency budget exceeded"
            await self.store.save_run(run)
        if run.cost_usd > run.budget.max_cost_usd:
            await self.store.cancel_run(run.id, "cost budget exceeded")
            run.status = RunStatus.BUDGET_EXCEEDED
            run.error = "cost budget exceeded"
            await self.store.save_run(run)
        return RunResponse(
            run=run,
            results=await self.store.list_results(run),
            trace_path=str(self.trace_path(run.id)),
        )

    async def cancel(self, run_id: str) -> bool:
        return await self.store.cancel_run(run_id)

    def trace_path(self, run_id: str) -> Path:
        return self.settings.trace_dir

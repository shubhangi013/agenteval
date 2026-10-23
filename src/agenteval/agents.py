from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from agenteval.config import Settings
from agenteval.llm import LLMClient
from agenteval.models import (
    CriticResult,
    Plan,
    RunState,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Verdict,
)
from agenteval.reliability import CircuitBreaker, resilient_call
from agenteval.store import RedisStore
from agenteval.tools import ToolType


class PlannerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def plan(self, run: RunState) -> Plan:
        tracer = trace.get_tracer("agenteval")
        with tracer.start_as_current_span("planner.decision") as span:
            span.set_attribute("run.id", run.id)
            plan = self._deterministic_plan(run)
            span.set_attribute("planner.task_count", len(plan.tasks))
            span.set_attribute("planner.strategy", "deterministic-tool-routing")
            return plan

    def _deterministic_plan(self, run: RunState) -> Plan:
        goal = run.goal.strip()
        calculator_match = re.search(r"(?:calculate|calculator)\s*:\s*(.+)", goal, re.I)
        python_match = re.search(r"(?:python|code)\s*:\s*(.+)", goal, re.I | re.S)
        fetch_match = re.search(r"(?:fetch|web)\s*:\s*(https?://\S+)", goal, re.I)
        if calculator_match:
            tool, tool_input = "calculator", {"expression": calculator_match.group(1)}
        elif python_match:
            tool, tool_input = "python_exec", {"code": python_match.group(1)}
        elif fetch_match:
            tool, tool_input = "web_fetch", {"url": fetch_match.group(1)}
        else:
            tool, tool_input = "python_exec", {"code": f"result = {goal!r}"}
        context = trace.get_current_span().get_span_context()
        task = TaskSpec(
            run_id=run.id,
            name="solve-goal",
            description=goal,
            tool=tool,
            tool_input=tool_input,
            acceptance_criteria="Tool returns a structured non-empty output without error.",
            parent_trace_id=format(context.trace_id, "032x") if context.is_valid else None,
            parent_span_id=format(context.span_id, "016x") if context.is_valid else None,
            failure_mode=str(run.metadata.get("failure_mode"))
            if run.metadata.get("failure_mode")
            else None,
        ).with_id()
        return Plan(run_id=run.id, tasks=[task])


class CriticAgent:
    async def evaluate(self, task: TaskSpec, result: TaskResult) -> CriticResult:
        with trace.get_tracer("agenteval").start_as_current_span("critic.verdict") as span:
            span.set_attribute("task.id", task.id)
            if result.error:
                verdict = (
                    Verdict.RETRY
                    if task.attempt + 1 < task.max_attempts
                    else Verdict.ESCALATE
                )
                score = 0.0
                reason = result.error
            elif not result.output:
                verdict, score, reason = Verdict.RETRY, 1.0, "empty structured output"
            else:
                verdict, score, reason = Verdict.ACCEPT, 5.0, "acceptance criteria satisfied"
            span.set_attribute("critic.verdict", verdict.value)
            span.set_attribute("critic.score", score)
            return CriticResult(task_id=task.id, verdict=verdict, score=score, reason=reason)


class WorkerAgent:
    def __init__(
        self,
        store: RedisStore,
        tools: dict[str, ToolType],
        settings: Settings,
        critic: CriticAgent,
    ) -> None:
        self.store = store
        self.tools = tools
        self.settings = settings
        self.critic = critic
        self.breakers = {
            name: CircuitBreaker(
                settings.breaker_failure_threshold,
                settings.breaker_window_s,
                settings.breaker_reset_s,
            )
            for name in tools
        }

    async def run_once(self) -> bool:
        task = await self.store.claim()
        if task is None:
            await self.store.recover_expired()
            return False
        await self.execute(task)
        return True

    async def execute(self, task: TaskSpec) -> None:
        tracer = trace.get_tracer("agenteval")
        started = time.monotonic()
        if await self.store.is_cancelled(task.run_id):
            await self.store.ack(
                TaskResult(
                    task_id=task.id,
                    run_id=task.run_id,
                    status=TaskStatus.CANCELLED,
                    attempt=task.attempt,
                )
            )
            return
        tool = self.tools.get(task.tool)
        if tool is None:
            await self._finish_error(task, f"unknown tool: {task.tool}", started)
            return
        try:
            cancel_event = asyncio.Event()

            async def watch_cancellation() -> None:
                while not cancel_event.is_set():
                    if await self.store.is_cancelled(task.run_id):
                        cancel_event.set()
                        return
                    await asyncio.sleep(0.1)

            cancellation_watcher = asyncio.create_task(watch_cancellation())
            parent_context = None
            if task.parent_trace_id and task.parent_span_id:
                remote = SpanContext(
                    trace_id=int(task.parent_trace_id, 16),
                    span_id=int(task.parent_span_id, 16),
                    is_remote=True,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                    trace_state=trace.DEFAULT_TRACE_STATE,
                )
                parent_context = trace.set_span_in_context(NonRecordingSpan(remote))
            with tracer.start_as_current_span("worker.task", context=parent_context) as task_span:
                task_span.set_attribute("run.id", task.run_id)
                task_span.set_attribute("task.id", task.id)
                task_span.set_attribute("task.attempt", task.attempt)
                validated = tool.input_schema.model_validate(task.tool_input)
                calls = 0

                async def invoke() -> Any:
                    nonlocal calls
                    calls += 1
                    with tracer.start_as_current_span("tool.call") as span:
                        span.set_attribute("tool.name", tool.name)
                        span.set_attribute("run.id", task.run_id)
                        span.set_attribute("task.id", task.id)
                        canonical_input = json.dumps(
                            validated.model_dump(mode="json"),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        span.set_attribute(
                            "tool.input_hash",
                            hashlib.sha256(canonical_input.encode()).hexdigest()[:16],
                        )
                        marker = f"agenteval:injected:{task.id}:{task.failure_mode}"
                        inject = bool(
                            task.failure_mode
                            and await self.store.redis.set(marker, "1", nx=True, ex=3600)
                        )
                        if task.failure_mode == "tool_timeout" and inject:
                            await asyncio.sleep(self.settings.tool_timeout_s * 2)
                        if task.failure_mode == "llm_429" and inject:
                            raise RuntimeError("LLM provider returned HTTP 429")
                        if task.failure_mode == "worker_crash" and inject:
                            await asyncio.sleep(10)
                        output = await tool.invoke(validated)  # type: ignore[arg-type]
                        return output.model_dump(mode="json")

                output = await resilient_call(
                    invoke,
                    name=tool.name,
                    timeout_s=self.settings.tool_timeout_s,
                    max_retries=self.settings.max_retries,
                    backoff_base_s=self.settings.backoff_base_s,
                    backoff_max_s=self.settings.backoff_max_s,
                    breaker=self.breakers[tool.name],
                    cancel_event=cancel_event,
                )
            result = TaskResult(
                task_id=task.id,
                run_id=task.run_id,
                status=TaskStatus.RUNNING,
                output=output,
                attempt=task.attempt,
                latency_ms=(time.monotonic() - started) * 1000,
            )
        except asyncio.CancelledError:
            result = TaskResult(
                task_id=task.id,
                run_id=task.run_id,
                status=TaskStatus.CANCELLED,
                attempt=task.attempt,
            )
        except Exception as exc:
            result = TaskResult(
                task_id=task.id,
                run_id=task.run_id,
                status=TaskStatus.RUNNING,
                error=str(exc),
                attempt=task.attempt,
                latency_ms=(time.monotonic() - started) * 1000,
            )
        finally:
            if "cancellation_watcher" in locals():
                cancellation_watcher.cancel()
        verdict = await self.critic.evaluate(task, result)
        if verdict.verdict == Verdict.ACCEPT:
            result.status = TaskStatus.ACCEPTED
            await self.store.ack(result)
        elif verdict.verdict == Verdict.RETRY:
            result.status = TaskStatus.RETRY
            await self.store.retry(task, result)
        else:
            result.status = TaskStatus.ESCALATED
            await self.store.ack(result)
        run = await self.store.get_run(task.run_id)
        if run:
            await self.store.release_ready_dependencies(run)
            await self.store.refresh_run_status(run.id)

    async def _finish_error(self, task: TaskSpec, error: str, started: float) -> None:
        result = TaskResult(
            task_id=task.id,
            run_id=task.run_id,
            status=TaskStatus.FAILED,
            error=error,
            attempt=task.attempt,
            latency_ms=(time.monotonic() - started) * 1000,
        )
        await self.store.ack(result)
        await self.store.refresh_run_status(task.run_id)

    async def serve(self) -> None:
        while True:
            await self.run_once()

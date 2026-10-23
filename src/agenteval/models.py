from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def deterministic_task_id(run_id: str, name: str, inputs: dict[str, Any]) -> str:
    canonical = json.dumps([run_id, name, inputs], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


class TaskStatus(StrEnum):
    BLOCKED = "blocked"
    QUEUED = "queued"
    RUNNING = "running"
    ACCEPTED = "accepted"
    RETRY = "retry"
    ESCALATED = "escalated"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Verdict(StrEnum):
    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"


class Budget(BaseModel):
    max_cost_usd: float = Field(gt=0)
    max_latency_ms: int = Field(gt=0)


class GoalRequest(BaseModel):
    goal: str = Field(min_length=1)
    max_cost_usd: float = Field(default=0.50, gt=0)
    max_latency_ms: int = Field(default=60_000, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    id: str = ""
    run_id: str
    name: str
    description: str
    tool: str
    tool_input: dict[str, Any]
    dependencies: list[str] = Field(default_factory=list)
    acceptance_criteria: str
    attempt: int = 0
    max_attempts: int = 3
    parent_trace_id: str | None = None
    parent_span_id: str | None = None
    failure_mode: str | None = None

    def with_id(self) -> TaskSpec:
        if self.id:
            return self
        return self.model_copy(
            update={"id": deterministic_task_id(self.run_id, self.name, self.tool_input)}
        )


class TaskResult(BaseModel):
    task_id: str
    run_id: str
    status: TaskStatus
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    attempt: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class CriticResult(BaseModel):
    task_id: str
    verdict: Verdict
    score: float = Field(ge=0, le=5)
    reason: str


class Plan(BaseModel):
    run_id: str
    tasks: list[TaskSpec]


class RunStatus(StrEnum):
    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"


class RunState(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    goal: str
    status: RunStatus = RunStatus.PLANNING
    budget: Budget
    cost_usd: float = 0.0
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    task_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    run: RunState
    results: list[TaskResult] = Field(default_factory=list)
    trace_path: str

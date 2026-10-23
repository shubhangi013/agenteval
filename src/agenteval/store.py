from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis

from agenteval.models import RunState, RunStatus, TaskResult, TaskSpec, TaskStatus

READY_QUEUE = "agenteval:queue:ready"
PROCESSING_QUEUE = "agenteval:queue:processing"
LEASES = "agenteval:leases"


class RedisStore:
    def __init__(self, redis: Redis, lease_s: float = 30.0) -> None:
        self.redis = redis
        self.lease_s = lease_s

    @staticmethod
    def run_key(run_id: str) -> str:
        return f"agenteval:run:{run_id}"

    @staticmethod
    def task_key(task_id: str) -> str:
        return f"agenteval:task:{task_id}"

    async def create_run(self, run: RunState) -> None:
        await self.redis.set(self.run_key(run.id), run.model_dump_json(), nx=True)

    async def get_run(self, run_id: str) -> RunState | None:
        raw = await self.redis.get(self.run_key(run_id))
        return RunState.model_validate_json(raw) if raw else None

    async def save_run(self, run: RunState) -> None:
        await self.redis.set(self.run_key(run.id), run.model_dump_json())

    async def enqueue(self, task: TaskSpec) -> bool:
        task = task.with_id()
        key = self.task_key(task.id)
        created = await self.redis.hsetnx(key, "spec", task.model_dump_json())
        if not created:
            return False
        status = TaskStatus.QUEUED if not task.dependencies else TaskStatus.BLOCKED
        await self.redis.hset(key, mapping={"status": status.value})
        if status == TaskStatus.QUEUED:
            await self.redis.lpush(READY_QUEUE, task.id)
        return True

    async def get_task(self, task_id: str) -> TaskSpec | None:
        raw = await self.redis.hget(self.task_key(task_id), "spec")
        return TaskSpec.model_validate_json(raw) if raw else None

    async def get_result(self, task_id: str) -> TaskResult | None:
        raw = await self.redis.hget(self.task_key(task_id), "result")
        return TaskResult.model_validate_json(raw) if raw else None

    async def claim(self, timeout_s: int = 1) -> TaskSpec | None:
        task_id = await self.redis.brpoplpush(READY_QUEUE, PROCESSING_QUEUE, timeout=timeout_s)
        if task_id is None:
            return None
        if isinstance(task_id, bytes):
            task_id = task_id.decode()
        task = await self.get_task(str(task_id))
        if task is None:
            await self.redis.lrem(PROCESSING_QUEUE, 1, task_id)
            return None
        await self.redis.hset(self.task_key(task.id), "status", TaskStatus.RUNNING.value)
        await self.redis.zadd(LEASES, {task.id: time.time() + self.lease_s})
        return task

    async def heartbeat(self, task_id: str) -> None:
        await self.redis.zadd(LEASES, {task_id: time.time() + self.lease_s})

    async def ack(self, result: TaskResult) -> None:
        key = self.task_key(result.task_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(
                key,
                mapping={"status": result.status.value, "result": result.model_dump_json()},
            )
            pipe.lrem(PROCESSING_QUEUE, 1, result.task_id)
            pipe.zrem(LEASES, result.task_id)
            await pipe.execute()

    async def retry(self, task: TaskSpec, result: TaskResult) -> None:
        updated = task.model_copy(update={"attempt": task.attempt + 1})
        key = self.task_key(task.id)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(
                key,
                mapping={
                    "spec": updated.model_dump_json(),
                    "status": TaskStatus.QUEUED.value,
                    "result": result.model_dump_json(),
                },
            )
            pipe.lrem(PROCESSING_QUEUE, 1, task.id)
            pipe.zrem(LEASES, task.id)
            pipe.lpush(READY_QUEUE, task.id)
            await pipe.execute()

    async def recover_expired(self) -> int:
        expired = await self.redis.zrangebyscore(LEASES, min=0, max=time.time())
        recovered = 0
        for raw_id in expired:
            task_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
            removed = await self.redis.zrem(LEASES, task_id)
            if removed:
                await self.redis.lrem(PROCESSING_QUEUE, 1, task_id)
                await self.redis.hset(self.task_key(task_id), "status", TaskStatus.QUEUED.value)
                await self.redis.lpush(READY_QUEUE, task_id)
                recovered += 1
        return recovered

    async def release_ready_dependencies(self, run: RunState) -> int:
        released = 0
        for task_id in run.task_ids:
            key = self.task_key(task_id)
            status = await self.redis.hget(key, "status")
            status_text = status.decode() if isinstance(status, bytes) else status
            if status_text != TaskStatus.BLOCKED.value:
                continue
            task = await self.get_task(task_id)
            if task is None:
                continue
            dependency_states = [
                await self.redis.hget(self.task_key(dep), "status") for dep in task.dependencies
            ]
            normalized = [
                value.decode() if isinstance(value, bytes) else value
                for value in dependency_states
            ]
            if all(value == TaskStatus.ACCEPTED.value for value in normalized):
                await self.redis.hset(key, "status", TaskStatus.QUEUED.value)
                await self.redis.lpush(READY_QUEUE, task_id)
                released += 1
        return released

    async def refresh_run_status(self, run_id: str) -> RunState | None:
        run = await self.get_run(run_id)
        if run is None:
            return None
        statuses = [
            await self.redis.hget(self.task_key(task_id), "status")
            for task_id in run.task_ids
        ]
        values = [value.decode() if isinstance(value, bytes) else value for value in statuses]
        terminal = {
            TaskStatus.ACCEPTED.value,
            TaskStatus.ESCALATED.value,
            TaskStatus.FAILED.value,
            TaskStatus.CANCELLED.value,
        }
        if values and all(value in terminal for value in values):
            failed = any(
                value in {TaskStatus.FAILED.value, TaskStatus.ESCALATED.value}
                for value in values
            )
            run.status = RunStatus.FAILED if failed else RunStatus.COMPLETED
            run.finished_at = datetime.now(UTC)
            await self.save_run(run)
        return run

    async def cancel_run(self, run_id: str, reason: str = "cancelled") -> bool:
        run = await self.get_run(run_id)
        if run is None:
            return False
        await self.redis.set(f"agenteval:cancel:{run_id}", reason)
        run.status = RunStatus.CANCELLED
        run.error = reason
        run.finished_at = datetime.now(UTC)
        await self.save_run(run)
        return True

    async def is_cancelled(self, run_id: str) -> bool:
        return bool(await self.redis.exists(f"agenteval:cancel:{run_id}"))

    async def add_cost(self, run_id: str, cost_usd: float) -> float:
        key = f"agenteval:cost:{run_id}"
        return float(await self.redis.incrbyfloat(key, cost_usd))

    async def list_results(self, run: RunState) -> list[TaskResult]:
        results: list[TaskResult] = []
        for task_id in run.task_ids:
            result = await self.get_result(task_id)
            if result:
                results.append(result)
        return results

    async def queue_stats(self) -> dict[str, Any]:
        return {
            "ready": await self.redis.llen(READY_QUEUE),
            "processing": await self.redis.llen(PROCESSING_QUEUE),
            "leased": await self.redis.zcard(LEASES),
        }

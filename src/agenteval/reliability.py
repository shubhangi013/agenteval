from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from opentelemetry import trace

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int
    window_s: float
    reset_s: float
    failures: deque[float] = field(default_factory=deque)
    opened_at: float | None = None

    def _prune(self, now: float) -> None:
        while self.failures and now - self.failures[0] > self.window_s:
            self.failures.popleft()

    def allow(self) -> bool:
        now = time.monotonic()
        if self.opened_at is None:
            return True
        if now - self.opened_at >= self.reset_s:
            self.opened_at = None
            self.failures.clear()
            return True
        return False

    def success(self) -> None:
        self.failures.clear()
        self.opened_at = None

    def failure(self) -> bool:
        now = time.monotonic()
        self._prune(now)
        self.failures.append(now)
        if len(self.failures) >= self.failure_threshold:
            self.opened_at = now
            return True
        return False


async def resilient_call(
    operation: Callable[[], Awaitable[T]],
    *,
    name: str,
    timeout_s: float,
    max_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
    breaker: CircuitBreaker,
    cancel_event: asyncio.Event | None = None,
) -> T:
    tracer = trace.get_tracer("agenteval")
    if not breaker.allow():
        with tracer.start_as_current_span("circuit_breaker.reject") as span:
            span.set_attribute("operation", name)
        raise CircuitOpenError(f"circuit open for {name}")

    for attempt in range(max_retries + 1):
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError
        try:
            with tracer.start_as_current_span("reliability.attempt") as span:
                span.set_attribute("operation", name)
                span.set_attribute("attempt", attempt)
                operation_task: asyncio.Future[T] = asyncio.ensure_future(operation())
                cancellation_task = (
                    asyncio.create_task(cancel_event.wait()) if cancel_event else None
                )
                wait_for: set[asyncio.Future[Any]] = {operation_task}
                if cancellation_task:
                    wait_for.add(cancellation_task)
                done, pending = await asyncio.wait(
                    wait_for,
                    timeout=timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for pending_task in pending:
                    pending_task.cancel()
                if not done:
                    raise TimeoutError(f"{name} timed out after {timeout_s}s")
                if cancellation_task and cancellation_task in done:
                    operation_task.cancel()
                    raise asyncio.CancelledError
                result: T = await operation_task
            breaker.success()
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            opened = breaker.failure()
            current = trace.get_current_span()
            current.add_event("operation.failure", {"attempt": attempt, "error": str(exc)})
            if opened:
                with tracer.start_as_current_span("circuit_breaker.trip") as span:
                    span.set_attribute("operation", name)
                    span.set_attribute("failures", breaker.failure_threshold)
            if attempt >= max_retries:
                raise
            delay = min(backoff_max_s, backoff_base_s * (2**attempt))
            delay *= 0.5 + random.random() * 0.5
            with tracer.start_as_current_span("retry.backoff") as span:
                span.set_attribute("operation", name)
                span.set_attribute("delay_s", delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")

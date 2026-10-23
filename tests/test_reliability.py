import asyncio

import pytest

from agenteval.reliability import CircuitBreaker, CircuitOpenError, resilient_call


@pytest.mark.asyncio
async def test_retry_recovers_transient_failure() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient")
        return "ok"

    result = await resilient_call(
        flaky,
        name="test",
        timeout_s=1,
        max_retries=1,
        backoff_base_s=0,
        backoff_max_s=0,
        breaker=CircuitBreaker(3, 10, 10),
    )
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold() -> None:
    breaker = CircuitBreaker(1, 10, 10)

    async def broken() -> None:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="down"):
        await resilient_call(
            broken,
            name="test",
            timeout_s=1,
            max_retries=0,
            backoff_base_s=0,
            backoff_max_s=0,
            breaker=breaker,
        )
    with pytest.raises(CircuitOpenError):
        await resilient_call(
            broken,
            name="test",
            timeout_s=1,
            max_retries=0,
            backoff_base_s=0,
            backoff_max_s=0,
            breaker=breaker,
        )


@pytest.mark.asyncio
async def test_cancellation_propagates_without_invocation() -> None:
    invoked = False
    cancel = asyncio.Event()
    cancel.set()

    async def operation() -> None:
        nonlocal invoked
        invoked = True

    with pytest.raises(asyncio.CancelledError):
        await resilient_call(
            operation,
            name="test",
            timeout_s=1,
            max_retries=1,
            backoff_base_s=0,
            backoff_max_s=0,
            breaker=CircuitBreaker(3, 10, 10),
            cancel_event=cancel,
        )
    assert not invoked

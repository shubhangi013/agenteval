from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from agenteval.bootstrap import create_runtime, create_store
from agenteval.config import Settings
from agenteval.models import GoalRequest, RunResponse
from agenteval.runtime import AgentRuntime
from agenteval.tracing import configure_tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    configure_tracing(settings.trace_dir / "spans.jsonl")
    store = create_store(settings)
    app.state.runtime = create_runtime(settings, store)
    app.state.store = store
    yield
    await store.redis.aclose()


app = FastAPI(title="AgentEval", version="0.1.0", lifespan=lifespan)


def runtime(request: Request) -> AgentRuntime:
    return request.app.state.runtime  # type: ignore[no-any-return]


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    store = request.app.state.store
    redis_ok = bool(await store.redis.ping())
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
        **await store.queue_stats(),
    }


@app.post("/v1/runs", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_run(payload: GoalRequest, request: Request) -> RunResponse:
    return await runtime(request).submit(payload)


@app.get("/v1/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, request: Request) -> RunResponse:
    response = await runtime(request).status(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="run not found")
    return response


@app.delete("/v1/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_run(run_id: str, request: Request) -> None:
    if not await runtime(request).cancel(run_id):
        raise HTTPException(status_code=404, detail="run not found")

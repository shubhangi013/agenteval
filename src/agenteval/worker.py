import asyncio
import os

from agenteval.bootstrap import create_store, create_worker
from agenteval.config import Settings
from agenteval.tracing import configure_tracing


async def serve() -> None:
    settings = Settings()
    worker_id = os.getenv("HOSTNAME", str(os.getpid()))
    configure_tracing(settings.trace_dir / f"spans-worker-{worker_id}.jsonl")
    store = create_store(settings)
    worker = create_worker(settings, store)
    try:
        await worker.serve()
    finally:
        await store.redis.aclose()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()

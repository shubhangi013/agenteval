"""Submit a goal and wait for its terminal result.

Run after `docker compose up` with: uv run python examples/submit_run.py
"""

import asyncio

import httpx


async def main() -> None:
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=30) as client:
        response = await client.post(
            "/v1/runs",
            json={
                "goal": "calculate: (17 * 19) + 4",
                "max_cost_usd": 0.05,
                "max_latency_ms": 10_000,
            },
        )
        response.raise_for_status()
        run_id = response.json()["run"]["id"]
        while True:
            state = (await client.get(f"/v1/runs/{run_id}")).json()
            if state["run"]["status"] != "running":
                print(state)
                return
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())

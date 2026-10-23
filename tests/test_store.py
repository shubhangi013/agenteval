import pytest

from agenteval.models import TaskSpec
from agenteval.store import READY_QUEUE, RedisStore


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.lists: dict[str, list[str]] = {}

    async def hsetnx(self, key: str, field: str, value: str) -> int:
        target = self.hashes.setdefault(key, {})
        if field in target:
            return 0
        target[field] = value
        return 1

    async def hset(self, key: str, *args: str, mapping: dict[str, str] | None = None) -> int:
        target = self.hashes.setdefault(key, {})
        if mapping:
            target.update(mapping)
        elif len(args) == 2:
            target[args[0]] = args[1]
        return 1

    async def lpush(self, key: str, value: str) -> int:
        target = self.lists.setdefault(key, [])
        target.insert(0, value)
        return len(target)


@pytest.mark.asyncio
async def test_duplicate_enqueue_is_noop() -> None:
    redis = FakeRedis()
    store = RedisStore(redis)  # type: ignore[arg-type]
    task = TaskSpec(
        run_id="run",
        name="task",
        description="test",
        tool="calculator",
        tool_input={"expression": "1+1"},
        acceptance_criteria="equals two",
    ).with_id()

    assert await store.enqueue(task)
    assert not await store.enqueue(task)
    assert redis.lists[READY_QUEUE] == [task.id]

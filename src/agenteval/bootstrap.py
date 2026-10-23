from redis.asyncio import Redis

from agenteval.agents import CriticAgent, PlannerAgent, WorkerAgent
from agenteval.config import Settings
from agenteval.llm import create_llm_client
from agenteval.runtime import AgentRuntime
from agenteval.store import RedisStore
from agenteval.tools import default_tools


def create_store(settings: Settings) -> RedisStore:
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return RedisStore(redis, lease_s=max(5.0, settings.tool_timeout_s * 2))


def create_runtime(settings: Settings, store: RedisStore) -> AgentRuntime:
    planner = PlannerAgent(create_llm_client(settings.llm_provider, settings))
    return AgentRuntime(store, planner, settings)


def create_worker(settings: Settings, store: RedisStore) -> WorkerAgent:
    return WorkerAgent(store, default_tools(settings), settings, CriticAgent())

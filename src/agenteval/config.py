from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTEVAL_", env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    llm_provider: Literal["mock", "openai", "azure_openai", "anthropic", "ollama"] = "mock"
    judge_provider: Literal["mock", "openai", "azure_openai", "anthropic", "ollama"] = "mock"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_input_usd_per_million: float = 0.15
    openai_output_usd_per_million: float = 0.60
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_input_usd_per_million: float = 0.0
    azure_openai_output_usd_per_million: float = 0.0
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-20241022"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_input_usd_per_million: float = 0.80
    anthropic_output_usd_per_million: float = 4.00
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    tool_timeout_s: float = 10.0
    max_retries: int = 3
    backoff_base_s: float = 0.1
    backoff_max_s: float = 4.0
    breaker_failure_threshold: int = 5
    breaker_window_s: float = 30.0
    breaker_reset_s: float = 15.0
    default_max_cost_usd: float = 0.50
    default_max_latency_ms: int = 60_000
    trace_dir: Path = Path("traces")
    inject_tool_timeout_rate: float = 0.0
    inject_llm_429_rate: float = 0.0


settings = Settings()

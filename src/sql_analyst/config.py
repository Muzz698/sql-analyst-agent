"""Application settings loaded from environment variables / .env.

All tunables live here so agents, the API, the UI and the eval harness share
one source of truth. Secrets are typed as ``SecretStr`` so they never leak into
logs or reprs.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM -----------------------------------------------------------------
    llm_provider: Literal["groq", "ollama", "fake"] = "groq"
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_model: str = "qwen3:4b"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.0

    # Retry with exponential backoff on rate limits / transient errors.
    llm_max_attempts: int = Field(default=5, ge=1)
    llm_backoff_initial_s: float = Field(default=2.0, gt=0)
    llm_backoff_max_s: float = Field(default=60.0, gt=0)

    # --- Database ------------------------------------------------------------
    db_path: Path = PROJECT_ROOT / "data" / "chinook.sqlite"
    checkpoint_db_path: Path = PROJECT_ROOT / "data" / "checkpoints.sqlite"
    query_timeout_ms: int = Field(default=5_000, gt=0)
    default_row_limit: int = Field(default=200, gt=0)

    # --- Agent behaviour -----------------------------------------------------
    max_sql_retries: int = Field(default=3, ge=0)
    use_few_shot: bool = True
    use_schema_pruning: bool = True
    require_approval_for_flagged: bool = True

    # --- Evaluation ----------------------------------------------------------
    eval_delay_s: float = Field(default=2.0, ge=0)

    # --- LangSmith (the SDK itself reads LANGSMITH_* from os.environ) ---------
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "sql-analyst-agent"


@lru_cache
def get_settings() -> Settings:
    return Settings()

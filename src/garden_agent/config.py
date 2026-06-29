"""Typed application settings loaded from the environment.

We use ``pydantic-settings`` so that:

* Missing required keys fail fast at startup, not deep in an adapter.
* Tests can override values by instantiating ``Settings(...)`` directly.
* No code outside this file reads ``os.environ`` — that is the only boundary.

Phase 1 only needs ``garden_data_path``. Phase 2 starts consuming the API key
fields, which are still ``Optional`` so unit tests (and ``--use-llm=false``
CLI runs) work without a real ``.env`` file. The adapter constructors are the
ones that fail loudly if a required key is missing — that keeps the boundary
in one obvious place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Phase 1 — local data
    garden_data_path: Path = Field(default=Path("data/garden.json"))

    # Phase 2 — external services (optional until then so tests don't choke)
    anthropic_api_key: str | None = None
    openweather_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 2048
    openweather_cache_ttl_seconds: int = 3600


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Module-level singleton.

    ``lru_cache`` keeps a single ``Settings`` instance for the whole process so
    repeated calls don't re-parse ``.env``. Tests can bypass it by passing an
    explicit ``Settings(...)`` to the code under test, or by calling
    ``get_settings.cache_clear()`` between cases.
    """
    return Settings()

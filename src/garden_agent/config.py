"""Typed application settings loaded from the environment.

We use ``pydantic-settings`` so that:

* Missing required keys fail fast at startup, not deep in an adapter.
* Tests can override values by instantiating ``Settings(...)`` directly.
* No code outside this file reads ``os.environ`` — that is the only boundary.

Phase 1 only needs ``garden_data_path``. Phase 2 will start consuming the
API key fields, which is why they are pre-declared as Optional now: that lets
Phase 1 tests run without a real ``.env`` file.
"""

from __future__ import annotations

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

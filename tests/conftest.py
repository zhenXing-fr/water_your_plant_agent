"""Shared test fixtures and **fake adapters**.

Why fakes instead of ``unittest.mock``? Three reasons that matter for agent code:

1. **Type-checkable** — fakes are real classes; mypy catches signature drift
   when a port evolves.
2. **Stateful but explicit** — they expose ``call_count`` / captured args, so
   tests assert behaviour, not "this method was called once with ANY".
3. **Reusable** — one fake serves every test in the suite, so test setup stays
   tiny and the same scenarios get exercised across layers.

The fakes satisfy the port Protocols **structurally** — they do NOT inherit
from them, which proves the port really is duck-typed.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from garden_agent.domain.models import (
    Garden,
    GrowthStage,
    Plant,
    SoilType,
    WeatherForecast,
)

# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------


class FakeWeatherAdapter:
    """In-memory WeatherPort that returns whatever forecasts you give it."""

    def __init__(self, forecasts: list[WeatherForecast]) -> None:
        self._forecasts = forecasts
        self.call_count = 0
        self.last_location: str | None = None
        self.last_days: int | None = None

    def get_forecast(self, location: str, days: int = 7) -> list[WeatherForecast]:
        self.call_count += 1
        self.last_location = location
        self.last_days = days
        return list(self._forecasts[:days])


class FakeGardenRepository:
    """In-memory GardenRepositoryPort backed by a single Garden instance."""

    def __init__(self, garden: Garden) -> None:
        self._garden = garden
        self.save_count = 0
        self.last_saved: Garden | None = None

    def get_garden(self, garden_id: str) -> Garden:
        if garden_id != self._garden.id:
            raise KeyError(garden_id)
        return self._garden

    def save_garden(self, garden: Garden) -> None:
        self.save_count += 1
        self.last_saved = garden
        self._garden = garden


class FakeLLMAdapter:
    """Returns a canned response. Phase 2 will give it tool-call behaviour."""

    def __init__(self, response: str = "") -> None:
        self._response = response
        self.call_count = 0
        self.last_prompt: str | None = None
        self.last_tools: list[dict[str, Any]] | None = None

    def generate(
        self,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_tools = tools
        return self._response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def week_start() -> date:
    return date(2026, 6, 29)  # a Monday


@pytest.fixture
def sample_plants() -> list[Plant]:
    return [
        Plant(
            name="Tom",
            plant_type="tomato",
            growth_stage=GrowthStage.ESTABLISHED,
            soil_type=SoilType.LOAMY,
        ),
        Plant(
            name="Lav-1",
            plant_type="lavender",
            growth_stage=GrowthStage.MATURE,
            soil_type=SoilType.SANDY,
        ),
        Plant(
            name="Basil-A",
            plant_type="basil",
            growth_stage=GrowthStage.SEEDLING,
            soil_type=SoilType.LOAMY,
        ),
    ]


@pytest.fixture
def sample_garden(sample_plants: list[Plant]) -> Garden:
    return Garden(id="g-001", location="Paris, France", plants=sample_plants)


@pytest.fixture
def sunny_week(week_start: date) -> list[WeatherForecast]:
    """Seven hot, dry days — every plant should be watered every day."""
    return [
        WeatherForecast(
            date=week_start + timedelta(days=i),
            temperature_max_c=28.0,
            temperature_min_c=18.0,
            precipitation_mm=0.0,
            humidity_percent=40.0,
            is_rain_expected=False,
        )
        for i in range(7)
    ]


@pytest.fixture
def mixed_week(week_start: date) -> list[WeatherForecast]:
    """Day 2 is a downpour; the rest are mild and dry."""
    forecasts = []
    for i in range(7):
        if i == 2:
            forecasts.append(
                WeatherForecast(
                    date=week_start + timedelta(days=i),
                    temperature_max_c=18.0,
                    temperature_min_c=12.0,
                    precipitation_mm=20.0,
                    humidity_percent=90.0,
                    is_rain_expected=True,
                )
            )
        else:
            forecasts.append(
                WeatherForecast(
                    date=week_start + timedelta(days=i),
                    temperature_max_c=22.0,
                    temperature_min_c=14.0,
                    precipitation_mm=0.0,
                    humidity_percent=50.0,
                    is_rain_expected=False,
                )
            )
    return forecasts

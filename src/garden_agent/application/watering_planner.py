"""Application service: orchestrate ports + domain to produce a WateringPlan.

This layer is allowed to import from ``ports`` and ``domain``, but NEVER from
``adapters``. That keeps it testable with the fakes in ``tests/conftest.py``
and makes adapter swaps (Phase 2 Claude -> Phase 7 LoRA) zero-cost.

Phase 1: only ``create_weekly_plan`` exists — pure deterministic planning that
exercises the domain rules. Phase 2 will add ``create_plan_with_llm`` which
runs the tool-calling loop against an :class:`LLMPort`.
"""

from __future__ import annotations

from datetime import date

from garden_agent.domain.models import WateringPlan
from garden_agent.domain.services import build_weekly_plan
from garden_agent.ports.garden import GardenRepositoryPort
from garden_agent.ports.llm import LLMPort
from garden_agent.ports.weather import WeatherPort


class WateringPlannerService:
    """Coordinator that pulls inputs from ports and delegates to the domain."""

    def __init__(
        self,
        weather: WeatherPort,
        garden_repo: GardenRepositoryPort,
        llm: LLMPort | None = None,
    ) -> None:
        # We store the LLM but do not use it in Phase 1. Phase 2 adds the
        # tool-calling method that consumes it.
        self._weather = weather
        self._garden_repo = garden_repo
        self._llm = llm

    def create_weekly_plan(self, garden_id: str, week_start: date) -> WateringPlan:
        """Deterministic plan built directly from the domain rules."""
        garden = self._garden_repo.get_garden(garden_id)
        forecasts = self._weather.get_forecast(garden.location, days=7)
        return build_weekly_plan(garden, forecasts, week_start)

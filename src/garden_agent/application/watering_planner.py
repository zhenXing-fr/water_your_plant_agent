"""Application service: orchestrate ports + domain to produce a WateringPlan.

This layer is allowed to import from ``ports`` and ``domain``, but NEVER from
``adapters``. That keeps it testable with the fakes in ``tests/conftest.py``
and makes adapter swaps (Phase 2 Claude -> Phase 7 LoRA) zero-cost.

Two entry points:

* :meth:`create_weekly_plan` — pure deterministic planning that exercises the
  domain rules directly. No LLM needed.
* :meth:`create_plan_with_llm` — drives a tool-calling loop against an
  :class:`LLMPort`. The LLM speaks a small JSON envelope (``tool_calls`` /
  ``final_plan``) that any adapter can produce, which is why the port stays
  ``generate(prompt, tools) -> str`` and provider-agnostic.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from garden_agent.application.tools import PLANNER_TOOLS, LLMResponseError
from garden_agent.domain.models import WateringPlan
from garden_agent.domain.services import build_weekly_plan
from garden_agent.ports.garden import GardenRepositoryPort
from garden_agent.ports.llm import LLMPort
from garden_agent.ports.weather import WeatherPort

MAX_TOOL_ITERATIONS: int = 6

SYSTEM_PROMPT: str = """\
You are a precision garden watering agent. Given a garden id and a week-start
date, you produce a 7-day WateringPlan in JSON.

Process:
1. Call get_garden_state to fetch the garden and its plants.
2. Call get_weather_forecast with the garden's location.
3. Emit the final plan.

Watering rules (apply exactly):
- Skip a day when precipitation_mm >= 10.0 (set skip_reason, leave actions empty).
- Water in the morning when temperature_max_c >= 25, otherwise evening.
- Base litres per growth stage: seedling 0.3, established 0.5, mature 0.8.
- Soil multiplier: sandy x1.40, well_draining x1.20, loamy x1.00, clay x0.70.
- Every WateringAction MUST include a specific reason explaining the choice.

Response protocol — respond with EXACTLY ONE JSON object, no prose, no fences:
  {"tool_calls": [{"id": "call_1", "name": "<tool>", "input": {...}}, ...]}
or
  {"final_plan": {
      "garden_id": "...", "week_start": "YYYY-MM-DD",
      "daily_plans": [
        {"date": "YYYY-MM-DD",
         "actions": [{"plant_name": "...", "amount_liters": 0.5,
                      "time_of_day": "morning", "reason": "..."}],
         "skip_reason": null}
      ]
  }}
"""


class WateringPlannerService:
    """Coordinator that pulls inputs from ports and delegates to the domain."""

    def __init__(
        self,
        weather: WeatherPort,
        garden_repo: GardenRepositoryPort,
        llm: LLMPort | None = None,
    ) -> None:
        self._weather = weather
        self._garden_repo = garden_repo
        self._llm = llm

    def create_weekly_plan(self, garden_id: str, week_start: date) -> WateringPlan:
        """Deterministic plan built directly from the domain rules."""
        garden = self._garden_repo.get_garden(garden_id)
        forecasts = self._weather.get_forecast(garden.location, days=7)
        return build_weekly_plan(garden, forecasts, week_start)

    # ------------------------------------------------------------------ #
    # LLM-driven planning                                                 #
    # ------------------------------------------------------------------ #

    def create_plan_with_llm(self, garden_id: str, week_start: date) -> WateringPlan:
        """Drive a tool-calling loop against the LLM port until it returns a plan."""
        if self._llm is None:
            raise RuntimeError("create_plan_with_llm requires an LLMPort; none was provided")

        transcript: list[str] = [
            f"SYSTEM:\n{SYSTEM_PROMPT}",
            f"USER:\n{self._initial_user_message(garden_id, week_start)}",
        ]

        for _ in range(MAX_TOOL_ITERATIONS):
            prompt = "\n\n".join(transcript)
            raw = self._llm.generate(prompt, tools=PLANNER_TOOLS)
            transcript.append(f"ASSISTANT:\n{raw}")

            envelope = self._parse_envelope(raw)

            if "final_plan" in envelope:
                return WateringPlan.model_validate(envelope["final_plan"])

            if "tool_calls" not in envelope:
                raise LLMResponseError(
                    "Response must contain either 'final_plan' or 'tool_calls'; "
                    f"got keys: {sorted(envelope)}"
                )

            for call in envelope["tool_calls"]:
                result = self._dispatch_tool(call)
                call_id = call.get("id") or call.get("name", "tool")
                transcript.append(
                    f"TOOL_RESULT [{call_id}]:\n{json.dumps(result, default=str, sort_keys=True)}"
                )

        raise LLMResponseError(
            f"LLM did not produce a final plan within {MAX_TOOL_ITERATIONS} iterations"
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _initial_user_message(garden_id: str, week_start: date) -> str:
        return (
            f'Plan a 7-day watering schedule for garden id "{garden_id}" '
            f"starting on {week_start.isoformat()}.\n"
            "Use the available tools to fetch the garden state and weather "
            "forecast, then return the final plan."
        )

    @staticmethod
    def _parse_envelope(raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"LLM did not return valid JSON: {raw!r}") from exc
        if not isinstance(data, dict):
            raise LLMResponseError(f"LLM response must be a JSON object, got {type(data).__name__}")
        return data

    def _dispatch_tool(self, call: dict[str, Any]) -> Any:
        name = call.get("name")
        args: dict[str, Any] = call.get("input") or {}

        if name == "get_weather_forecast":
            location = args["location"]
            days = int(args.get("days", 7))
            forecasts = self._weather.get_forecast(location, days)
            return [f.model_dump(mode="json") for f in forecasts]

        if name == "get_garden_state":
            garden = self._garden_repo.get_garden(args["garden_id"])
            return garden.model_dump(mode="json")

        raise LLMResponseError(f"Unknown tool requested by LLM: {name!r}")

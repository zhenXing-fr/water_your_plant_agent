"""Application-layer tests for the LLM-driven planning entry point.

The point of these tests is to pin the tool-calling loop contract WITHOUT
hitting any real LLM. We script ``FakeLLMAdapter`` with the JSON envelope the
planner expects and assert that:

* both ports are dispatched when the LLM asks for the matching tool
* the final ``WateringPlan`` JSON is parsed back into a domain object
* every error path raises ``LLMResponseError`` (and not a generic exception)

The fake plays the role any future adapter (Claude in Phase 2, LoRA in Phase 7)
will have to satisfy structurally — proving the port boundary really holds.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from garden_agent.application.tools import (
    GET_GARDEN_TOOL,
    GET_WEATHER_TOOL,
    LLMResponseError,
)
from garden_agent.application.watering_planner import (
    MAX_TOOL_ITERATIONS,
    WateringPlannerService,
)
from garden_agent.domain.models import Garden, WateringPlan, WeatherForecast
from tests.conftest import FakeGardenRepository, FakeLLMAdapter, FakeWeatherAdapter

# ---------------------------------------------------------------------------
# Helpers — keep the JSON envelopes the LLM is expected to emit out of the
# tests' assertion bodies so each test reads as a scenario, not a fixture pile.
# ---------------------------------------------------------------------------


def _tool_call_envelope(*calls: dict[str, Any]) -> str:
    return json.dumps({"tool_calls": list(calls)})


def _final_plan_envelope(plan: WateringPlan) -> str:
    return json.dumps({"final_plan": plan.model_dump(mode="json")})


def _golden_plan(
    garden: Garden, forecasts: list[WeatherForecast], week_start: date
) -> WateringPlan:
    """Use the pure-domain planner to build a 'golden' plan the fake LLM will echo."""
    from garden_agent.domain.services import build_weekly_plan

    return build_weekly_plan(garden, forecasts, week_start)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_create_plan_with_llm_dispatches_both_tools_then_returns_final_plan(
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
    week_start: date,
) -> None:
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    expected_plan = _golden_plan(sample_garden, sunny_week, week_start)

    scripted = [
        # Turn 1: LLM asks for the garden first.
        _tool_call_envelope(
            {"id": "c1", "name": "get_garden_state", "input": {"garden_id": sample_garden.id}}
        ),
        # Turn 2: armed with the garden, asks for weather at its location.
        _tool_call_envelope(
            {
                "id": "c2",
                "name": "get_weather_forecast",
                "input": {"location": sample_garden.location, "days": 7},
            }
        ),
        # Turn 3: emits the final plan.
        _final_plan_envelope(expected_plan),
    ]
    llm = FakeLLMAdapter(response=scripted)

    service = WateringPlannerService(weather=weather, garden_repo=garden_repo, llm=llm)
    plan = service.create_plan_with_llm(sample_garden.id, week_start)

    # Ports were actually exercised — not just the LLM.
    assert garden_repo._garden.id == sample_garden.id  # noqa: SLF001 — internal check ok in tests
    assert weather.call_count == 1
    assert weather.last_location == sample_garden.location
    assert weather.last_days == 7

    # LLM saw the tool catalogue on every call.
    assert llm.call_count == 3
    for tools in llm.tools_history:
        assert tools is not None
        names = {t["name"] for t in tools}
        assert names == {GET_WEATHER_TOOL["name"], GET_GARDEN_TOOL["name"]}

    # The plan is parsed back into a real domain object, equal to the golden one.
    assert isinstance(plan, WateringPlan)
    assert plan == expected_plan


def test_create_plan_with_llm_accepts_immediate_final_plan(
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
    week_start: date,
) -> None:
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    expected_plan = _golden_plan(sample_garden, sunny_week, week_start)

    llm = FakeLLMAdapter(response=[_final_plan_envelope(expected_plan)])
    service = WateringPlannerService(weather=weather, garden_repo=garden_repo, llm=llm)

    plan = service.create_plan_with_llm(sample_garden.id, week_start)

    assert plan == expected_plan
    assert llm.call_count == 1
    # No tools invoked because the LLM never asked.
    assert weather.call_count == 0


def test_tool_results_appear_in_subsequent_prompts(
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
    week_start: date,
) -> None:
    """Each follow-up prompt must include the previous TOOL_RESULT block,
    otherwise the LLM has no memory between turns."""
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    expected_plan = _golden_plan(sample_garden, sunny_week, week_start)

    scripted = [
        _tool_call_envelope(
            {"id": "c1", "name": "get_garden_state", "input": {"garden_id": sample_garden.id}}
        ),
        _final_plan_envelope(expected_plan),
    ]
    llm = FakeLLMAdapter(response=scripted)
    service = WateringPlannerService(weather=weather, garden_repo=garden_repo, llm=llm)
    service.create_plan_with_llm(sample_garden.id, week_start)

    second_prompt = llm.prompts[1]
    assert "TOOL_RESULT [c1]" in second_prompt
    assert sample_garden.location in second_prompt  # garden state was injected


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_create_plan_with_llm_requires_an_llm_port(
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
    week_start: date,
) -> None:
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    service = WateringPlannerService(weather=weather, garden_repo=garden_repo)

    with pytest.raises(RuntimeError, match="LLMPort"):
        service.create_plan_with_llm(sample_garden.id, week_start)


def test_invalid_json_response_raises_llm_response_error(
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
    week_start: date,
) -> None:
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    llm = FakeLLMAdapter(response=["not json at all"])
    service = WateringPlannerService(weather=weather, garden_repo=garden_repo, llm=llm)

    with pytest.raises(LLMResponseError, match="valid JSON"):
        service.create_plan_with_llm(sample_garden.id, week_start)


def test_non_object_json_response_raises_llm_response_error(
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
    week_start: date,
) -> None:
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    llm = FakeLLMAdapter(response=[json.dumps(["just", "a", "list"])])
    service = WateringPlannerService(weather=weather, garden_repo=garden_repo, llm=llm)

    with pytest.raises(LLMResponseError, match="JSON object"):
        service.create_plan_with_llm(sample_garden.id, week_start)


def test_unknown_envelope_keys_raise_llm_response_error(
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
    week_start: date,
) -> None:
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    llm = FakeLLMAdapter(response=[json.dumps({"something_else": 42})])
    service = WateringPlannerService(weather=weather, garden_repo=garden_repo, llm=llm)

    with pytest.raises(LLMResponseError, match="final_plan.*tool_calls|tool_calls.*final_plan"):
        service.create_plan_with_llm(sample_garden.id, week_start)


def test_unknown_tool_name_raises_llm_response_error(
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
    week_start: date,
) -> None:
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    llm = FakeLLMAdapter(
        response=[_tool_call_envelope({"id": "x", "name": "delete_database", "input": {}})]
    )
    service = WateringPlannerService(weather=weather, garden_repo=garden_repo, llm=llm)

    with pytest.raises(LLMResponseError, match="Unknown tool"):
        service.create_plan_with_llm(sample_garden.id, week_start)


def test_runaway_tool_loop_raises_after_max_iterations(
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
    week_start: date,
) -> None:
    """An LLM that keeps calling tools forever must be cut off, not crash the host."""
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)

    looping_call = _tool_call_envelope(
        {"id": "loop", "name": "get_garden_state", "input": {"garden_id": sample_garden.id}}
    )
    llm = FakeLLMAdapter(response=[looping_call] * MAX_TOOL_ITERATIONS)
    service = WateringPlannerService(weather=weather, garden_repo=garden_repo, llm=llm)

    with pytest.raises(LLMResponseError, match=f"{MAX_TOOL_ITERATIONS} iterations"):
        service.create_plan_with_llm(sample_garden.id, week_start)
    assert llm.call_count == MAX_TOOL_ITERATIONS

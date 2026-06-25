"""Application-layer tests for :class:`WateringPlannerService`.

The point of these tests is to prove the service wires ports correctly
WITHOUT touching the network. We use the fakes from ``conftest.py``.
"""

from __future__ import annotations

from datetime import timedelta

from garden_agent.application.watering_planner import WateringPlannerService
from tests.conftest import (
    FakeGardenRepository,
    FakeLLMAdapter,
    FakeWeatherAdapter,
)


def test_create_weekly_plan_pulls_from_both_ports(sample_garden, sunny_week, week_start) -> None:
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    service = WateringPlannerService(weather=weather, garden_repo=garden_repo)

    plan = service.create_weekly_plan(sample_garden.id, week_start)

    # Weather port called exactly once with the garden's location
    assert weather.call_count == 1
    assert weather.last_location == sample_garden.location
    assert weather.last_days == 7

    # 7-day plan, no skips on a sunny week, every plant covered each day
    assert plan.garden_id == sample_garden.id
    assert plan.week_start == week_start
    assert len(plan.daily_plans) == 7
    for daily in plan.daily_plans:
        assert daily.skip_reason is None
        assert {a.plant_name for a in daily.actions} == {p.name for p in sample_garden.plants}


def test_unknown_garden_id_raises_keyerror(sample_garden, sunny_week, week_start) -> None:
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    service = WateringPlannerService(weather=weather, garden_repo=garden_repo)

    try:
        service.create_weekly_plan("does-not-exist", week_start)
    except KeyError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("expected KeyError for unknown garden id")


def test_planner_accepts_optional_llm_port(sample_garden, sunny_week, week_start) -> None:
    # In Phase 1 the LLM is optional and unused; this test pins that contract
    # so Phase 2 can extend, not redesign.
    weather = FakeWeatherAdapter(forecasts=sunny_week)
    garden_repo = FakeGardenRepository(garden=sample_garden)
    llm = FakeLLMAdapter(response="unused")

    service = WateringPlannerService(weather=weather, garden_repo=garden_repo, llm=llm)
    plan = service.create_weekly_plan(sample_garden.id, week_start)

    assert llm.call_count == 0
    assert len(plan.daily_plans) == 7
    assert plan.daily_plans[-1].date == week_start + timedelta(days=6)

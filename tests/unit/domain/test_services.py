"""Tests for the pure watering rules."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from garden_agent.domain.models import (
    GrowthStage,
    Plant,
    SoilType,
    WeatherForecast,
)
from garden_agent.domain.services import (
    BASE_LITERS_BY_STAGE,
    HOT_DAY_THRESHOLD_C,
    RAIN_SKIP_THRESHOLD_MM,
    SOIL_MULTIPLIER,
    build_daily_plan,
    build_weekly_plan,
    compute_water_amount,
    preferred_time_of_day,
    rain_covers_watering,
)


def _forecast(d: date, *, rain: float = 0.0, tmax: float = 22.0) -> WeatherForecast:
    return WeatherForecast(
        date=d,
        temperature_max_c=tmax,
        temperature_min_c=tmax - 10,
        precipitation_mm=rain,
        humidity_percent=50.0,
        is_rain_expected=rain > 0,
    )


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def test_rain_covers_watering_when_at_or_above_threshold() -> None:
    assert rain_covers_watering(_forecast(date(2026, 6, 29), rain=RAIN_SKIP_THRESHOLD_MM))
    assert rain_covers_watering(_forecast(date(2026, 6, 29), rain=25.0))


def test_rain_does_not_cover_watering_below_threshold() -> None:
    assert not rain_covers_watering(_forecast(date(2026, 6, 29), rain=9.9))


def test_preferred_time_of_day_is_morning_on_hot_days() -> None:
    assert (
        preferred_time_of_day(_forecast(date(2026, 6, 29), tmax=HOT_DAY_THRESHOLD_C)) == "morning"
    )
    assert preferred_time_of_day(_forecast(date(2026, 6, 29), tmax=30.0)) == "morning"


def test_preferred_time_of_day_is_evening_on_mild_days() -> None:
    assert preferred_time_of_day(_forecast(date(2026, 6, 29), tmax=20.0)) == "evening"


# ---------------------------------------------------------------------------
# Water amount table — one assertion per (stage, soil) pair so failures are obvious.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", list(GrowthStage))
@pytest.mark.parametrize("soil", list(SoilType))
def test_compute_water_amount_matches_base_times_multiplier(
    stage: GrowthStage, soil: SoilType
) -> None:
    plant = Plant(name="x", plant_type="t", growth_stage=stage, soil_type=soil)
    expected = round(BASE_LITERS_BY_STAGE[stage] * SOIL_MULTIPLIER[soil], 2)
    assert compute_water_amount(plant) == expected


# ---------------------------------------------------------------------------
# Daily / weekly composition
# ---------------------------------------------------------------------------


def test_daily_plan_skips_on_rainy_day(sample_plants: list[Plant]) -> None:
    plan = build_daily_plan(_forecast(date(2026, 6, 29), rain=20.0), sample_plants)
    assert plan.actions == []
    assert plan.skip_reason is not None
    assert "20.0mm" in plan.skip_reason


def test_daily_plan_waters_every_plant_when_dry(sample_plants: list[Plant]) -> None:
    plan = build_daily_plan(_forecast(date(2026, 6, 29), rain=0.0, tmax=28.0), sample_plants)
    assert plan.skip_reason is None
    assert [a.plant_name for a in plan.actions] == [p.name for p in sample_plants]
    assert all(a.time_of_day == "morning" for a in plan.actions)
    assert all(a.reason for a in plan.actions)


def test_weekly_plan_uses_only_forecasts_within_the_week(sample_garden, week_start: date) -> None:
    # Provide 10 days — only 7 should make it into the plan
    forecasts = [_forecast(week_start + timedelta(days=i)) for i in range(-1, 9)]
    plan = build_weekly_plan(sample_garden, forecasts, week_start)

    dates = [dp.date for dp in plan.daily_plans]
    assert dates == [week_start + timedelta(days=i) for i in range(7)]


def test_weekly_plan_skips_rainy_day_in_mixed_week(sample_garden, mixed_week, week_start) -> None:
    plan = build_weekly_plan(sample_garden, mixed_week, week_start)
    skipped = [dp for dp in plan.daily_plans if dp.skip_reason]
    watered = [dp for dp in plan.daily_plans if not dp.skip_reason]
    assert len(skipped) == 1
    assert skipped[0].date == week_start + timedelta(days=2)
    assert len(watered) == 6

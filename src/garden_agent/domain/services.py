"""Pure domain logic for watering decisions.

This module has ZERO external dependencies (no httpx, no anthropic, no chromadb).
Everything here can be unit-tested with plain Python — that is the whole point of
keeping the domain isolated. Adapters and the application layer call into these
functions; the LLM (Phase 2+) is later asked to reproduce / extend this reasoning.

Watering rules (single source of truth — also pinned in agent_code.md):

* Skip a day when ``precipitation_mm >= 10.0`` (rain does the watering for us).
* Water in the morning when ``temperature_max_c >= 25``; otherwise evening
  (cooler temps mean less evaporation, so evening is fine).
* Base amount per growth stage:
    - seedling     -> 0.3 L
    - established  -> 0.5 L
    - mature       -> 0.8 L
* Soil multiplier (applied to the base amount):
    - sandy          -> 1.40  (drains fast, needs more)
    - well_draining  -> 1.20
    - loamy          -> 1.00  (reference)
    - clay           -> 0.70  (holds water, needs less)
"""

from __future__ import annotations

from datetime import date, timedelta

from garden_agent.domain.models import (
    DailyPlan,
    Garden,
    GrowthStage,
    Plant,
    SoilType,
    WateringAction,
    WateringPlan,
    WeatherForecast,
)

# ---------------------------------------------------------------------------
# Constants — exposed so tests can reference them rather than magic numbers.
# ---------------------------------------------------------------------------

RAIN_SKIP_THRESHOLD_MM: float = 10.0
HOT_DAY_THRESHOLD_C: float = 25.0

BASE_LITERS_BY_STAGE: dict[GrowthStage, float] = {
    GrowthStage.SEEDLING: 0.3,
    GrowthStage.ESTABLISHED: 0.5,
    GrowthStage.MATURE: 0.8,
}

SOIL_MULTIPLIER: dict[SoilType, float] = {
    SoilType.SANDY: 1.40,
    SoilType.WELL_DRAINING: 1.20,
    SoilType.LOAMY: 1.00,
    SoilType.CLAY: 0.70,
}


# ---------------------------------------------------------------------------
# Pure predicates and calculators.
# ---------------------------------------------------------------------------


def rain_covers_watering(forecast: WeatherForecast) -> bool:
    """Return True when the forecasted rainfall makes watering redundant."""
    return forecast.precipitation_mm >= RAIN_SKIP_THRESHOLD_MM


def preferred_time_of_day(forecast: WeatherForecast) -> str:
    """Return ``"morning"`` on hot days, otherwise ``"evening"``."""
    return "morning" if forecast.temperature_max_c >= HOT_DAY_THRESHOLD_C else "evening"


def compute_water_amount(plant: Plant) -> float:
    """Litres for a single plant, rounded to two decimals."""
    base = BASE_LITERS_BY_STAGE[plant.growth_stage]
    multiplier = SOIL_MULTIPLIER[plant.soil_type]
    return round(base * multiplier, 2)


# ---------------------------------------------------------------------------
# Composers — they assemble the small predicates above into domain objects.
# ---------------------------------------------------------------------------


def _reason_for(plant: Plant, forecast: WeatherForecast) -> str:
    return (
        f"{plant.growth_stage.value} {plant.plant_type} in {plant.soil_type.value} soil; "
        f"forecast {forecast.temperature_max_c:.0f}\u00b0C max, "
        f"{forecast.precipitation_mm:.1f}mm rain"
    )


def build_daily_plan(forecast: WeatherForecast, plants: list[Plant]) -> DailyPlan:
    """Produce one DailyPlan for one forecast day."""
    if rain_covers_watering(forecast):
        return DailyPlan(
            date=forecast.date,
            actions=[],
            skip_reason=(
                f"rain forecast {forecast.precipitation_mm:.1f}mm "
                f">= {RAIN_SKIP_THRESHOLD_MM}mm threshold"
            ),
        )

    time_of_day = preferred_time_of_day(forecast)
    actions = [
        WateringAction(
            plant_name=plant.name,
            amount_liters=compute_water_amount(plant),
            time_of_day=time_of_day,
            reason=_reason_for(plant, forecast),
        )
        for plant in plants
    ]
    return DailyPlan(date=forecast.date, actions=actions, skip_reason=None)


def build_weekly_plan(
    garden: Garden,
    forecasts: list[WeatherForecast],
    week_start: date,
) -> WateringPlan:
    """Assemble a 7-day WateringPlan from a garden and its forecasts.

    Forecasts that fall outside ``[week_start, week_start + 6 days]`` are ignored;
    missing days simply produce no DailyPlan entry. The caller is responsible for
    providing the right slice of forecasts — this function does not invent days.
    """
    week_end = week_start + timedelta(days=6)
    in_week = [f for f in forecasts if week_start <= f.date <= week_end]
    in_week.sort(key=lambda f: f.date)

    daily_plans = [build_daily_plan(f, garden.plants) for f in in_week]
    return WateringPlan(
        garden_id=garden.id,
        week_start=week_start,
        daily_plans=daily_plans,
    )

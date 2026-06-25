"""Smoke tests on the Pydantic domain models.

These guard against subtle regressions like loosening ``frozen=True`` or
renaming a field — both of which would silently break the LLM prompts later.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from garden_agent.domain.models import (
    GrowthStage,
    Plant,
    SoilType,
    WateringAction,
    WateringPlan,
)


def test_plant_defaults_apply_when_optional_fields_omitted() -> None:
    p = Plant(
        name="Tom",
        plant_type="tomato",
        growth_stage=GrowthStage.ESTABLISHED,
        soil_type=SoilType.LOAMY,
    )

    assert p.last_watered is None
    assert p.notes == ""


def test_growth_stage_serialises_as_string() -> None:
    # The (str, Enum) base means JSON-dumping gives plain strings, which is
    # what the LLM will see and produce.
    assert GrowthStage.MATURE.value == "mature"


def test_watering_action_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        WateringAction(
            plant_name="Tom",
            amount_liters=0.5,
            time_of_day="morning",
            # reason missing on purpose
        )  # type: ignore[call-arg]


def test_watering_plan_round_trips_through_json() -> None:
    plan = WateringPlan(garden_id="g-001", week_start=date(2026, 6, 29), daily_plans=[])
    dumped = plan.model_dump_json()
    restored = WateringPlan.model_validate_json(dumped)
    assert restored == plan

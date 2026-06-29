"""CLI smoke tests.

We deliberately do NOT call ``OpenWeather`` or ``Anthropic`` here — instead
we monkeypatch the composition root's ``_build_service`` to return a service
wired to the in-memory fakes. That keeps CLI tests fast, offline, and
focused on argument parsing + output formatting (the actual responsibility
of ``main.py``).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from garden_agent import main as cli
from garden_agent.application.watering_planner import WateringPlannerService
from garden_agent.domain.models import Garden, WeatherForecast
from tests.conftest import FakeGardenRepository, FakeWeatherAdapter


@pytest.fixture
def wire_fake_service(
    monkeypatch: pytest.MonkeyPatch,
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
) -> WateringPlannerService:
    """Replace ``_build_service`` with one that returns a fake-wired service."""
    service = WateringPlannerService(
        weather=FakeWeatherAdapter(forecasts=sunny_week),
        garden_repo=FakeGardenRepository(garden=sample_garden),
    )
    monkeypatch.setattr(cli, "_build_service", lambda settings, use_llm: service)
    return service


def test_next_monday_skips_to_following_monday() -> None:
    # 2026-06-29 is a Monday — next "next Monday" should be the same day.
    assert cli._next_monday(date(2026, 6, 29)) == date(2026, 6, 29)
    # Tuesday → following Monday.
    assert cli._next_monday(date(2026, 6, 30)) == date(2026, 7, 6)


def test_cli_prints_json_plan_for_pure_domain_path(
    capsys: pytest.CaptureFixture[str],
    wire_fake_service: WateringPlannerService,
    sample_garden: Garden,
) -> None:
    exit_code = cli.main(["--garden-id", sample_garden.id, "--week-start", "2026-06-29"])
    assert exit_code == 0

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["garden_id"] == sample_garden.id
    assert payload["week_start"] == "2026-06-29"
    assert len(payload["daily_plans"]) == 7


def test_cli_defaults_week_start_to_next_monday(
    capsys: pytest.CaptureFixture[str],
    wire_fake_service: WateringPlannerService,
    sample_garden: Garden,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Freeze "today" to a known Tuesday so the default branch is deterministic.
    class _FrozenDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 6, 30)

    monkeypatch.setattr(cli, "date", _FrozenDate)

    cli.main(["--garden-id", sample_garden.id])
    payload = json.loads(capsys.readouterr().out)
    assert payload["week_start"] == "2026-07-06"  # the following Monday


def test_cli_missing_required_arg_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main([])  # no --garden-id


def test_build_service_requires_openweather_key_for_pure_domain(
    tmp_path: Path,
) -> None:
    settings = cli.Settings(
        garden_data_path=tmp_path / "garden.json",
        openweather_api_key=None,
    )
    with pytest.raises(SystemExit, match="OPENWEATHER_API_KEY"):
        cli._build_service(settings, use_llm=False)


def test_build_service_requires_both_keys_for_llm(tmp_path: Path) -> None:
    settings = cli.Settings(
        garden_data_path=tmp_path / "garden.json",
        openweather_api_key=None,
        anthropic_api_key="sk-ant-x",
    )
    with pytest.raises(SystemExit, match="OPENWEATHER_API_KEY"):
        cli._build_service(settings, use_llm=True)

    settings2 = cli.Settings(
        garden_data_path=tmp_path / "garden.json",
        openweather_api_key="ow-x",
        anthropic_api_key=None,
    )
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        cli._build_service(settings2, use_llm=True)


def test_build_service_pure_domain_returns_service_when_key_set(
    tmp_path: Path,
) -> None:
    settings = cli.Settings(
        garden_data_path=tmp_path / "garden.json",
        openweather_api_key="ow-x",
    )
    service = cli._build_service(settings, use_llm=False)
    assert isinstance(service, WateringPlannerService)


def test_build_service_llm_returns_service_when_keys_set(tmp_path: Path) -> None:
    settings = cli.Settings(
        garden_data_path=tmp_path / "garden.json",
        openweather_api_key="ow-x",
        anthropic_api_key="sk-ant-x",
    )
    service = cli._build_service(settings, use_llm=True)
    assert isinstance(service, WateringPlannerService)


def test_cli_use_llm_branch_calls_llm_method(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    sample_garden: Garden,
    sunny_week: list[WeatherForecast],
    week_start: date,
) -> None:
    from garden_agent.domain.services import build_weekly_plan
    from tests.conftest import FakeLLMAdapter

    expected = build_weekly_plan(sample_garden, sunny_week, week_start)
    scripted = [json.dumps({"final_plan": expected.model_dump(mode="json")})]
    service = WateringPlannerService(
        weather=FakeWeatherAdapter(forecasts=sunny_week),
        garden_repo=FakeGardenRepository(garden=sample_garden),
        llm=FakeLLMAdapter(response=scripted),
    )
    monkeypatch.setattr(cli, "_build_service", lambda settings, use_llm: service)

    code = cli.main(
        ["--garden-id", sample_garden.id, "--week-start", week_start.isoformat(), "--use-llm"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == expected.model_dump(mode="json")

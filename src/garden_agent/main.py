"""Command-line entry point — the **composition root** of the application.

================================================================================
Tomorrow's reading order: claude.py → main.py (this file).
================================================================================

What a "composition root" is
----------------------------
Mark Seemann coined the term: it's the single place in your program where you
new up every concrete dependency and wire them together. Everything else in the
codebase only ever sees ports (abstract interfaces). The composition root is
where the abstract plan ("we need a Weather, a Repo, and an LLM") becomes
concrete ("OpenWeatherAdapter, JSONGardenRepository, ClaudeAdapter").

Why have one at all? Because it keeps the rest of the code testable. The
``WateringPlannerService`` doesn't import any adapter — it can't even *see*
``anthropic`` or ``httpx``. Tests inject fake ports; ``main.py`` injects the
real ones. Same service code, different wiring.

Read more:
    https://blog.ploeh.dk/2011/07/28/CompositionRoot/

What this file does, in three lines
-----------------------------------
1. Parse CLI flags (``--garden-id``, ``--week-start``, ``--use-llm``).
2. Build the three concrete adapters from settings (the *only* file that
   instantiates them).
3. Hand them to ``WateringPlannerService`` and print the resulting plan.

Usage::

    # Pure domain rules (no LLM, deterministic, just needs OpenWeather):
    uv run python -m garden_agent.main --garden-id g-001

    # Same but pick a week:
    uv run python -m garden_agent.main --garden-id g-001 --week-start 2026-07-06

    # Claude-driven plan (uses the tool-calling loop):
    uv run python -m garden_agent.main --garden-id g-001 --use-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date, timedelta

# Adapter imports are concentrated HERE — this is the one place we allow
# infrastructure imports to leak. The application/, domain/, ports/ layers
# never see anthropic / httpx / file I/O.
from garden_agent.adapters.garden.json_repo import JSONGardenRepository
from garden_agent.adapters.llm.claude import ClaudeAdapter
from garden_agent.adapters.weather.openweather import OpenWeatherAdapter
from garden_agent.application.watering_planner import WateringPlannerService
from garden_agent.config import Settings, get_settings
from garden_agent.domain.models import WateringPlan


def _next_monday(today: date) -> date:
    """Default ``week_start`` if the user didn't pass one.

    ``date.weekday()`` returns 0 for Monday … 6 for Sunday. So:

    * Monday  → ``weekday()`` = 0 → ``(7 - 0) % 7 = 0`` → today.
    * Tuesday → ``weekday()`` = 1 → ``(7 - 1) % 7 = 6`` → today + 6 days = Mon.
    * Sunday  → ``weekday()`` = 6 → ``(7 - 6) % 7 = 1`` → tomorrow.

    The ``% 7`` is the trick: it folds the "Monday case" back to 0 instead of 7.
    """
    days_ahead = (7 - today.weekday()) % 7
    return today + timedelta(days=days_ahead)


def _build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface.

    Why ``argparse`` and not ``click`` or ``typer``? Standard library, zero
    extra dependencies, fine for one command. We'll graduate to ``typer`` if
    sub-commands ever appear (Phase 4+ when LangGraph adds nodes).
    """
    parser = argparse.ArgumentParser(
        prog="garden-agent",
        description="Produce a 7-day watering plan for a garden.",
    )
    parser.add_argument(
        "--garden-id",
        required=True,
        help="Garden identifier (e.g. g-001)",
    )
    parser.add_argument(
        "--week-start",
        # argparse will pass the user's string straight into this function and
        # use the return value as ``args.week_start``. ``date.fromisoformat``
        # raises ``ValueError`` on bad input → argparse turns that into a nice
        # CLI error message automatically. Free input validation.
        type=date.fromisoformat,
        default=None,
        help="Week start date in YYYY-MM-DD (defaults to next Monday)",
    )
    parser.add_argument(
        "--use-llm",
        # store_true: presence of the flag → True, absence → False.
        action="store_true",
        help="Drive the plan through Claude instead of the pure domain rules.",
    )
    return parser


def _build_service(settings: Settings, *, use_llm: bool) -> WateringPlannerService:
    """The actual composition root.

    Reads typed settings, instantiates the right adapters, and returns a
    ready-to-use service. Two branches:

    * ``--use-llm`` ⇒ need both OpenWeather AND Anthropic keys (Claude drives
      the tool-calling loop and asks for weather through the tools).
    * Pure domain ⇒ only OpenWeather (Claude is not involved at all).

    Validation lives here, not in ``main``: keeping it close to the
    construction means a clearer "this is why I refused to start" message.
    """
    garden_repo = JSONGardenRepository(settings.garden_data_path)

    if use_llm:
        # Fail-fast: surface missing config BEFORE building any client. We
        # raise ``SystemExit`` (not ``LLMError`` / ``WeatherError``) because
        # this is a user-facing CLI condition, not a runtime adapter failure
        # — argparse-style exits print the message and return a non-zero
        # exit code, which is exactly the UX we want at the boundary.
        if not settings.openweather_api_key:
            raise SystemExit("OPENWEATHER_API_KEY is required for --use-llm")
        if not settings.anthropic_api_key:
            raise SystemExit("ANTHROPIC_API_KEY is required for --use-llm")

        weather = OpenWeatherAdapter(
            api_key=settings.openweather_api_key,
            cache_ttl_seconds=settings.openweather_cache_ttl_seconds,
        )
        llm = ClaudeAdapter(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=settings.claude_max_tokens,
        )
        return WateringPlannerService(
            weather=weather,
            garden_repo=garden_repo,
            llm=llm,
        )

    # Pure-domain path: still needs a real forecast (the rules in
    # ``domain/services.py`` consume ``WeatherForecast`` objects either way).
    if not settings.openweather_api_key:
        raise SystemExit(
            "OPENWEATHER_API_KEY is required to fetch a real forecast. "
            "Set it in .env or pass --use-llm with a configured Claude key."
        )
    weather = OpenWeatherAdapter(
        api_key=settings.openweather_api_key,
        cache_ttl_seconds=settings.openweather_cache_ttl_seconds,
    )
    # No ``llm=`` argument → the service falls into deterministic mode.
    return WateringPlannerService(weather=weather, garden_repo=garden_repo)


def main(argv: Sequence[str] | None = None) -> int:
    """Program entry point.

    ``argv`` is accepted as a parameter (rather than read implicitly from
    ``sys.argv``) so tests can call ``main(["--garden-id", "g-001"])`` directly
    — see ``tests/unit/test_main.py``. When ``None``, argparse falls back to
    ``sys.argv[1:]`` which is the real CLI behaviour.

    Returns an int exit code: 0 = success. We don't ``sys.exit`` here so the
    tests can assert the return value without catching ``SystemExit``.
    """
    args = _build_parser().parse_args(argv)
    # ``or`` short-circuits: if the user passed --week-start it wins, otherwise
    # fall back to next Monday based on the current date.
    week_start = args.week_start or _next_monday(date.today())

    settings = get_settings()
    service = _build_service(settings, use_llm=args.use_llm)

    if args.use_llm:
        plan: WateringPlan = service.create_plan_with_llm(args.garden_id, week_start)
    else:
        plan = service.create_weekly_plan(args.garden_id, week_start)

    # Serialise the plan as nicely-formatted JSON to stdout.
    # ``mode="json"`` makes pydantic convert dates → strings, enums → values,
    # so the output is portable JSON (not Python literals).
    json.dump(plan.model_dump(mode="json"), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via `python -m`
    raise SystemExit(main())

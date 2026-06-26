"""Tool schemas the planner advertises to any LLM port.

These dicts are intentionally provider-agnostic in *shape* but follow Anthropic's
``tools`` schema format (``name`` / ``description`` / ``input_schema``) because
that is the first concrete adapter we will ship (Phase 2). Other adapters
(OpenAI, MCP, LoRA) can translate from this representation in their own layer
— the application never has to care.

The tools themselves are dispatched in
:meth:`garden_agent.application.watering_planner.WateringPlannerService._dispatch_tool`,
which is the single place that maps a tool name to a port call.
"""

from __future__ import annotations

from typing import Any

GET_WEATHER_TOOL: dict[str, Any] = {
    "name": "get_weather_forecast",
    "description": "Get the multi-day weather forecast for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'Paris, France'",
            },
            "days": {"type": "integer", "default": 7},
        },
        "required": ["location"],
    },
}

GET_GARDEN_TOOL: dict[str, Any] = {
    "name": "get_garden_state",
    "description": "Retrieve the current garden state including all plants.",
    "input_schema": {
        "type": "object",
        "properties": {
            "garden_id": {"type": "string"},
        },
        "required": ["garden_id"],
    },
}

PLANNER_TOOLS: list[dict[str, Any]] = [GET_WEATHER_TOOL, GET_GARDEN_TOOL]


class LLMResponseError(RuntimeError):
    """The LLM response could not be parsed or violated the response contract."""

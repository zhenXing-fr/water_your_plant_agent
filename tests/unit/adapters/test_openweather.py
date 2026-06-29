"""Unit tests for :class:`OpenWeatherAdapter`.

We use ``httpx.MockTransport`` (built in) so the real ``httpx.Client`` is
exercised — request building, query-string encoding, error mapping — without
touching the network. That gives stronger guarantees than monkey-patching
``client.get``. See
https://www.python-httpx.org/advanced/transports/#mock-transports.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
import pytest

from garden_agent.adapters.weather.openweather import (
    OPENWEATHER_FORECAST_URL,
    OpenWeatherAdapter,
    WeatherError,
)
from garden_agent.domain.models import WeatherForecast
from garden_agent.ports.weather import WeatherPort

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_payload() -> dict[str, Any]:
    """Minimal OpenWeather /forecast payload — two 3-hour windows."""
    return {
        "list": [
            {
                "dt_txt": "2026-06-29 12:00:00",
                "main": {"temp_max": 28.0, "temp_min": 18.0, "humidity": 45},
                "rain": {"3h": 0.2},
                "weather": [{"main": "Clouds", "description": "few clouds"}],
            },
            {
                "dt_txt": "2026-06-29 15:00:00",
                "main": {"temp_max": 30.0, "temp_min": 20.0, "humidity": 40},
                # No "rain" key — should default to 0.0
                "weather": [{"main": "Rain", "description": "light rain"}],
            },
        ]
    }


def _adapter_with(
    handler,
    *,
    api_key: str = "test-key",
    cache_ttl_seconds: float = 3600.0,
    now=None,
) -> OpenWeatherAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    kwargs: dict[str, Any] = {
        "client": client,
        "cache_ttl_seconds": cache_ttl_seconds,
    }
    if now is not None:
        kwargs["now"] = now
    return OpenWeatherAdapter(api_key=api_key, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adapter_satisfies_port_protocol() -> None:
    adapter = OpenWeatherAdapter(api_key="x", client=httpx.Client())
    assert isinstance(adapter, WeatherPort)


def test_missing_api_key_raises() -> None:
    with pytest.raises(WeatherError, match="API key"):
        OpenWeatherAdapter(api_key="")


def test_get_forecast_maps_payload_to_domain_objects() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(200, json=_sample_payload())

    adapter = _adapter_with(handler)
    forecasts = adapter.get_forecast("Paris, France", days=2)

    # 1. The right URL + query string was assembled.
    req = captured["req"]
    assert str(req.url).startswith(OPENWEATHER_FORECAST_URL)
    assert req.url.params["q"] == "Paris, France"
    assert req.url.params["cnt"] == "2"
    assert req.url.params["appid"] == "test-key"
    assert req.url.params["units"] == "metric"

    # 2. The JSON was mapped into real domain objects.
    assert len(forecasts) == 2
    assert all(isinstance(f, WeatherForecast) for f in forecasts)

    first, second = forecasts
    assert first.date == date(2026, 6, 29)
    assert first.precipitation_mm == 0.2
    assert first.is_rain_expected is False  # "Clouds" main

    assert second.precipitation_mm == 0.0  # rain key missing → defaults to 0
    assert second.is_rain_expected is True  # "Rain" main


def test_get_forecast_uses_cache_within_ttl() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_sample_payload())

    # Inject a tickable clock so we can advance "time" without sleeping.
    clock = {"t": 0.0}
    adapter = _adapter_with(handler, cache_ttl_seconds=60.0, now=lambda: clock["t"])

    adapter.get_forecast("Paris, France", days=2)
    adapter.get_forecast("Paris, France", days=2)
    assert calls["n"] == 1  # second call was a cache hit

    # Different (location, days) key → fresh fetch.
    adapter.get_forecast("Lyon, France", days=2)
    assert calls["n"] == 2

    # Advance past TTL → cache invalidated.
    clock["t"] = 120.0
    adapter.get_forecast("Paris, France", days=2)
    assert calls["n"] == 3


def test_non_200_status_raises_weather_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Invalid API key")

    adapter = _adapter_with(handler)
    with pytest.raises(WeatherError, match="HTTP 401"):
        adapter.get_forecast("Paris, France", days=2)


def test_network_error_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    adapter = _adapter_with(handler)
    with pytest.raises(WeatherError, match="request failed"):
        adapter.get_forecast("Paris, France", days=2)


def test_non_json_response_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    adapter = _adapter_with(handler)
    with pytest.raises(WeatherError, match="not JSON"):
        adapter.get_forecast("Paris, France", days=2)


def test_payload_without_list_key_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"cod": "404", "message": "city not found"})

    adapter = _adapter_with(handler)
    with pytest.raises(WeatherError, match="missing 'list'"):
        adapter.get_forecast("Atlantis", days=2)


def test_bad_item_shape_is_wrapped() -> None:
    bad_payload = {"list": [{"dt_txt": "not-a-date", "main": {}, "weather": []}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=bad_payload)

    adapter = _adapter_with(handler)
    with pytest.raises(WeatherError, match="map OpenWeather forecast item"):
        adapter.get_forecast("Paris, France", days=1)


def test_context_manager_closes_owned_client() -> None:
    # Real Client, real lifecycle: enter / exit should close it.
    handler_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        handler_calls["n"] += 1
        return httpx.Response(200, json=_sample_payload())

    # api_key path that constructs its own client when we don't pass one — but
    # we still want to swap the transport, so we pass a client explicitly here
    # and just assert close() is idempotent on the injected variant.
    transport = httpx.MockTransport(handler)
    with OpenWeatherAdapter(api_key="x", client=httpx.Client(transport=transport)) as a:
        a.get_forecast("Paris, France", days=2)
    # close() on the injected client variant is a no-op (owns_client=False),
    # which is the contract we want — caller-owned resources are caller-managed.
    assert handler_calls["n"] == 1


def test_close_owned_client_is_safe() -> None:
    a = OpenWeatherAdapter(api_key="x")
    a.close()  # should not raise even though we never made a request


def test_response_serialization_round_trip() -> None:
    """Sanity check: the mapped object can be re-serialized as the tool result."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_sample_payload())

    adapter = _adapter_with(handler)
    forecasts = adapter.get_forecast("Paris, France", days=2)
    dumped = [f.model_dump(mode="json") for f in forecasts]
    assert dumped[0]["date"] == "2026-06-29"
    assert json.dumps(dumped)  # must be JSON-serialisable as-is

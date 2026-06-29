"""OpenWeather adapter — concrete :class:`WeatherPort` over httpx.

Design notes you should internalise (this is the *teaching* adapter):

1. **Dependency injection beats globals.** We accept an optional
   ``httpx.Client`` so tests can drive the adapter with
   :class:`httpx.MockTransport` instead of monkey-patching the network.
   See https://www.python-httpx.org/advanced/transports/#mock-transports.

2. **Caching belongs in the adapter, not the application.** The application
   asks the port for a forecast; how often that hits the wire is an
   infrastructure concern. A tiny in-process TTL cache (keyed by
   ``(location, days)``) is enough for the agent loop, which may call the
   tool several times within one planning session.

3. **No raw exceptions leak.** Every HTTP/parsing failure is wrapped in
   :class:`WeatherError` so the application layer can catch one type instead
   of three different libraries' error hierarchies. This is the classic
   "translate at the boundary" pattern.

4. **Domain mapping happens here too.** The adapter is the only place that
   knows OpenWeather's JSON shape; the application only ever sees
   :class:`WeatherForecast` instances. Read:
   https://martinfowler.com/eaaCatalog/dataMapper.html
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

import httpx
from pydantic import ValidationError

from garden_agent.domain.models import WeatherForecast

OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
DEFAULT_TIMEOUT_SECONDS = 10.0


class WeatherError(RuntimeError):
    """Raised on any HTTP, network or schema failure when fetching weather."""


class OpenWeatherAdapter:
    """Calls the OpenWeather "5 day / 3 hour" forecast endpoint.

    Note: the upstream endpoint returns one entry per 3-hour window, not per
    day. Per the project spec, ``days`` is forwarded as OpenWeather's ``cnt``
    (item count) and each returned item is mapped 1:1 into a
    :class:`WeatherForecast`. Aggregating to a real per-day view is left as a
    later refinement so this slice stays minimal.
    """

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        cache_ttl_seconds: float = 3600.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise WeatherError("OpenWeather API key is required")
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)
        self._owns_client = client is None
        self._cache_ttl = cache_ttl_seconds
        # ``now`` is injected so tests can advance "time" without sleeping.
        self._now = now
        self._cache: dict[tuple[str, int], tuple[float, list[WeatherForecast]]] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the underlying client if we created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenWeatherAdapter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Port surface                                                        #
    # ------------------------------------------------------------------ #

    def get_forecast(self, location: str, days: int = 7) -> list[WeatherForecast]:
        key = (location, days)
        cached = self._cache.get(key)
        if cached is not None:
            inserted_at, forecasts = cached
            if (self._now() - inserted_at) < self._cache_ttl:
                return list(forecasts)

        params = {
            "q": location,
            "cnt": days,
            "appid": self._api_key,
            "units": "metric",
        }

        try:
            response = self._client.get(OPENWEATHER_FORECAST_URL, params=params)
        except httpx.HTTPError as exc:
            raise WeatherError(f"OpenWeather request failed: {exc}") from exc

        if response.status_code != 200:
            raise WeatherError(
                f"OpenWeather returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise WeatherError("OpenWeather response was not JSON") from exc

        forecasts = self._map_payload(payload)
        self._cache[key] = (self._now(), forecasts)
        return list(forecasts)

    # ------------------------------------------------------------------ #
    # Internal mapping                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _map_payload(payload: Mapping[str, Any]) -> list[WeatherForecast]:
        items = payload.get("list")
        if not isinstance(items, list):
            raise WeatherError(
                "OpenWeather payload missing 'list' array: "
                f"keys={sorted(payload) if isinstance(payload, Mapping) else 'n/a'}"
            )

        forecasts: list[WeatherForecast] = []
        for raw in items:
            try:
                forecasts.append(OpenWeatherAdapter._map_item(raw))
            except (KeyError, TypeError, ValueError, ValidationError) as exc:
                raise WeatherError(f"Could not map OpenWeather forecast item: {raw!r}") from exc
        return forecasts

    @staticmethod
    def _map_item(item: Mapping[str, Any]) -> WeatherForecast:
        main = item["main"]
        weather_blocks = item.get("weather") or []
        return WeatherForecast(
            date=datetime.fromisoformat(item["dt_txt"]).date(),
            temperature_max_c=float(main["temp_max"]),
            temperature_min_c=float(main["temp_min"]),
            precipitation_mm=float(item.get("rain", {}).get("3h", 0.0)),
            humidity_percent=float(main["humidity"]),
            is_rain_expected=any(
                "rain" in (block.get("main", "")).lower() for block in weather_blocks
            ),
        )

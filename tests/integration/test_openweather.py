"""Live OpenWeather API check — skipped in CI."""

from __future__ import annotations

import pytest

from garden_agent.adapters.weather.openweather import OpenWeatherAdapter
from garden_agent.config import get_settings
from garden_agent.domain.models import WeatherForecast


@pytest.mark.integration
def test_openweather_returns_real_forecast() -> None:
    settings = get_settings()
    if not settings.openweather_api_key:
        pytest.skip("OPENWEATHER_API_KEY not set")

    with OpenWeatherAdapter(api_key=settings.openweather_api_key) as adapter:
        forecasts = adapter.get_forecast("Paris, France", days=4)

    assert forecasts, "Expected at least one forecast item"
    assert all(isinstance(f, WeatherForecast) for f in forecasts)

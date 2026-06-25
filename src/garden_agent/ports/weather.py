"""Weather port — what the application asks from any weather source.

The port is a ``Protocol`` (PEP 544), not an ABC. Adapters do NOT inherit from
this class; they only need to expose the same method signatures (structural
subtyping). This keeps adapters free of any import of the port and lets us
plug in tests' fakes without ceremony.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from garden_agent.domain.models import WeatherForecast


@runtime_checkable
class WeatherPort(Protocol):
    """Fetch a multi-day weather forecast for a free-form location string."""

    def get_forecast(self, location: str, days: int = 7) -> list[WeatherForecast]:
        """Return one WeatherForecast per day, ordered ascending by date.

        Implementations may raise their own adapter-specific error (e.g.
        ``WeatherError``) on transport / parsing failures. The application
        layer should treat any exception as a degraded run.
        """
        ...

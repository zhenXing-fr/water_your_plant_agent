"""Integration tests that hit real external services.

Skipped in CI via the ``integration`` marker. Run locally with::

    uv run pytest -m integration

You must export ``OPENWEATHER_API_KEY`` (or have it in ``.env``).
"""

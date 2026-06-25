"""Garden repository port — read and persist Garden aggregates."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from garden_agent.domain.models import Garden


@runtime_checkable
class GardenRepositoryPort(Protocol):
    """Load and save :class:`Garden` aggregates by id."""

    def get_garden(self, garden_id: str) -> Garden:
        """Return the Garden with the given id.

        Implementations MUST raise ``KeyError`` if the id is unknown.
        """
        ...

    def save_garden(self, garden: Garden) -> None:
        """Persist the Garden (full replacement, not a patch)."""
        ...

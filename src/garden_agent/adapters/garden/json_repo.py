"""File-backed implementation of :class:`GardenRepositoryPort`.

Why a separate file repository at all in an agent project? Because the *port*
exists for testability: today we read a single JSON file, tomorrow we may swap
in a Postgres or DynamoDB adapter — the application service doesn't change.
This module is therefore the canonical example of an "adapter": it has the
only ``open()``/``json.load()`` calls in the codebase, and everywhere else
stays pure.

Behaviour contract (re-stated from the port docstring, plus file specifics):

* :meth:`get_garden` raises :class:`KeyError` when the requested id does not
  match the stored garden — this matches the in-memory fake in
  ``tests/conftest.py``, so application-level tests stay portable.
* :meth:`save_garden` writes the **entire** aggregate atomically: we write to
  a temp file and ``os.replace`` it onto the target. ``os.replace`` is atomic
  on POSIX and Windows (since Python 3.3), so a crash mid-write can never leave
  a half-written ``garden.json``. This is the standard "atomic write" pattern
  — see https://docs.python.org/3/library/os.html#os.replace.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from garden_agent.domain.models import Garden


class GardenRepositoryError(RuntimeError):
    """Raised when the underlying JSON file is unreadable or malformed."""


class JSONGardenRepository:
    """Read/write a single :class:`Garden` aggregate from a JSON file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------ #
    # Port surface                                                        #
    # ------------------------------------------------------------------ #

    def get_garden(self, garden_id: str) -> Garden:
        if not self._path.exists():
            raise GardenRepositoryError(f"Garden data file not found: {self._path}")
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GardenRepositoryError(
                f"Garden data file is not valid JSON: {self._path}"
            ) from exc

        garden = Garden.model_validate(data)
        if garden.id != garden_id:
            raise KeyError(garden_id)
        return garden

    def save_garden(self, garden: Garden) -> None:
        """Atomically replace the stored garden."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = garden.model_dump(mode="json")

        # Write to a sibling temp file in the same directory so os.replace is
        # cross-filesystem-safe, then atomically swap it in.
        fd, tmp_name = tempfile.mkstemp(
            prefix=".garden-", suffix=".json.tmp", dir=str(self._path.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, sort_keys=True)
                fp.write("\n")
            os.replace(tmp_path, self._path)
        except Exception:
            # Best-effort cleanup if anything went wrong before the rename.
            tmp_path.unlink(missing_ok=True)
            raise

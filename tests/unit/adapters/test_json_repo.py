"""Unit tests for :class:`JSONGardenRepository`.

Why use ``tmp_path`` instead of mocking the filesystem? Because the adapter's
entire reason for existing is to talk to the filesystem — mocking ``open()``
would test the mock, not the code. ``tmp_path`` is pytest's built-in
per-test directory (cleaned up automatically); see
https://docs.pytest.org/en/stable/how-to/tmp_path.html.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from garden_agent.adapters.garden.json_repo import (
    GardenRepositoryError,
    JSONGardenRepository,
)
from garden_agent.domain.models import Garden, GrowthStage, Plant, SoilType
from garden_agent.ports.garden import GardenRepositoryPort


def _write_garden(path: Path, garden: Garden) -> None:
    path.write_text(json.dumps(garden.model_dump(mode="json"), indent=2), encoding="utf-8")


@pytest.fixture
def garden() -> Garden:
    return Garden(
        id="g-test",
        location="Lyon, France",
        plants=[
            Plant(
                name="Tom",
                plant_type="tomato",
                growth_stage=GrowthStage.ESTABLISHED,
                soil_type=SoilType.LOAMY,
            )
        ],
    )


def test_repository_satisfies_port_protocol(tmp_path: Path) -> None:
    repo = JSONGardenRepository(tmp_path / "garden.json")
    # @runtime_checkable Protocol — proves structural conformance at import time.
    assert isinstance(repo, GardenRepositoryPort)


def test_get_garden_returns_parsed_aggregate(tmp_path: Path, garden: Garden) -> None:
    path = tmp_path / "garden.json"
    _write_garden(path, garden)
    repo = JSONGardenRepository(path)

    loaded = repo.get_garden(garden.id)

    assert loaded == garden
    # Domain objects are frozen so equality is structural — that's the proof
    # that pydantic round-tripped every field correctly.


def test_get_garden_raises_key_error_when_id_mismatches(tmp_path: Path, garden: Garden) -> None:
    path = tmp_path / "garden.json"
    _write_garden(path, garden)
    repo = JSONGardenRepository(path)

    with pytest.raises(KeyError):
        repo.get_garden("does-not-exist")


def test_get_garden_raises_when_file_missing(tmp_path: Path) -> None:
    repo = JSONGardenRepository(tmp_path / "missing.json")
    with pytest.raises(GardenRepositoryError, match="not found"):
        repo.get_garden("anything")


def test_get_garden_raises_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "garden.json"
    path.write_text("{not json", encoding="utf-8")
    repo = JSONGardenRepository(path)

    with pytest.raises(GardenRepositoryError, match="not valid JSON"):
        repo.get_garden("anything")


def test_save_garden_writes_round_trippable_json(tmp_path: Path, garden: Garden) -> None:
    path = tmp_path / "garden.json"
    repo = JSONGardenRepository(path)

    repo.save_garden(garden)

    # Round-trip: load the written file with a fresh repo and compare.
    reloaded = JSONGardenRepository(path).get_garden(garden.id)
    assert reloaded == garden


def test_save_garden_creates_parent_directories(tmp_path: Path, garden: Garden) -> None:
    nested = tmp_path / "deep" / "nested" / "garden.json"
    repo = JSONGardenRepository(nested)

    repo.save_garden(garden)

    assert nested.exists()
    assert json.loads(nested.read_text(encoding="utf-8"))["id"] == garden.id


def test_save_garden_overwrites_atomically_and_leaves_no_tempfiles(
    tmp_path: Path, garden: Garden
) -> None:
    path = tmp_path / "garden.json"
    repo = JSONGardenRepository(path)
    repo.save_garden(garden)

    mutated = garden.model_copy(update={"location": "Marseille, France"})
    repo.save_garden(mutated)

    # No stray ".garden-*.tmp" files left behind by the atomic write.
    leftovers = list(tmp_path.glob(".garden-*.tmp"))
    assert leftovers == []
    assert repo.get_garden(garden.id).location == "Marseille, France"

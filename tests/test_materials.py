"""
test_materials.py
===================

Unit tests for `ntp_campus.materials.group_buildings`, which takes the
flat list produced by `load_building_dimensions` (one dict per .ply file)
and groups it by `base_name` so that e.g. a building's `_brick` and
`_concrete` parts end up in the same group.

We build the input dicts by hand here instead of calling
`load_building_dimensions` itself, so these tests don't depend on any
.ply files being present on disk -- `group_buildings` is a pure
dict-grouping function and can be tested in isolation.

Run with:
    pytest tests/test_materials.py -v
"""

import pytest

from ntp_campus.materials import group_buildings


def _fake_building_record(filename, base_name, **overrides):
    """Minimal stand-in for one entry of `load_building_dimensions()`'s output."""
    record = {
        "filename": filename,
        "base_name": base_name,
        "width": 10.0, "depth": 10.0, "height": 5.0,
        "center_x": 0.0, "center_y": 0.0, "center_z": 2.5,
        "min_x": -5.0, "min_y": -5.0, "min_z": 0.0,
        "max_x": 5.0, "max_y": 5.0, "max_z": 5.0,
        "num_vertices": 8,
    }
    record.update(overrides)
    return record


def test_group_buildings_groups_multiple_material_parts_together():
    building_data = [
        _fake_building_record("BuildingA_brick.ply", "BuildingA"),
        _fake_building_record("BuildingA_concrete.ply", "BuildingA"),
        _fake_building_record("BuildingB_concrete.ply", "BuildingB"),
    ]

    groups = group_buildings(building_data)

    assert set(groups.keys()) == {"BuildingA", "BuildingB"}
    assert len(groups["BuildingA"]) == 2
    assert len(groups["BuildingB"]) == 1


def test_group_buildings_preserves_original_records():
    building_data = [_fake_building_record("BuildingC_brick.ply", "BuildingC", width=42.0)]

    groups = group_buildings(building_data)

    assert groups["BuildingC"][0]["width"] == 42.0
    assert groups["BuildingC"][0]["filename"] == "BuildingC_brick.ply"


def test_group_buildings_handles_empty_input():
    assert group_buildings([]) == {}


def test_group_buildings_single_part_building():
    building_data = [_fake_building_record("BuildingD_concrete.ply", "BuildingD")]

    groups = group_buildings(building_data)

    assert list(groups.keys()) == ["BuildingD"]
    assert len(groups["BuildingD"]) == 1

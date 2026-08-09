"""
test_antenna_pattern.py
=========================

Unit tests for `ntp_campus.antenna_pattern`:
  - `compass_azimuth_to_yaw`: compass (0deg = north, clockwise) -> Sionna yaw
    (0deg = +X, counter-clockwise) conversion.
  - `SectorAntennaSpec`: total_tilt_deg = electrical + mechanical tilt.

NOTE: importing `ntp_campus.antenna_pattern` pulls in `mitsuba`, `drjit`
and `sionna.rt` at module level (needed for `register_antenna_pattern`
and the `air3278_broadcast_v_pattern` ray-tracing callback), so these
packages must be installed (see requirements.txt) to run this file --
same as for running the simulation itself.

Run with:
    pytest tests/test_antenna_pattern.py -v
"""

import math

import numpy as np
import pytest

from ntp_campus.antenna_pattern import compass_azimuth_to_yaw, SectorAntennaSpec


# =========================================================================
# compass_azimuth_to_yaw
# =========================================================================
# We always pass yaw_offset_deg explicitly (instead of relying on the
# yaw_offset_deg=None -> config.YAW_OFFSET_DEG default) so these tests
# don't depend on whether some other test/module has already mutated the
# shared `config` module (e.g. via antenna_pattern.apply_antenna_config).

@pytest.mark.parametrize(
    "azimuth_deg, expected_yaw_rad",
    [
        (0.0, math.pi / 2),     # north   -> Sionna's +Y direction (90deg math convention)
        (90.0, 0.0),            # east    -> Sionna's +X direction (0deg math convention)
        (180.0, 3 * math.pi / 2),  # south -> 270deg math convention
        (270.0, math.pi),       # west    -> 180deg math convention
    ],
)
def test_compass_azimuth_to_yaw_cardinal_directions(azimuth_deg, expected_yaw_rad):
    yaw = compass_azimuth_to_yaw(azimuth_deg, yaw_offset_deg=0.0)
    assert yaw == pytest.approx(expected_yaw_rad, abs=1e-9)


def test_compass_azimuth_to_yaw_is_periodic():
    """Azimuths 360 degrees apart must map to the same yaw (mod 2*pi)."""
    yaw_a = compass_azimuth_to_yaw(45.0, yaw_offset_deg=0.0)
    yaw_b = compass_azimuth_to_yaw(45.0 + 360.0, yaw_offset_deg=0.0)
    assert yaw_a == pytest.approx(yaw_b % (2 * math.pi), abs=1e-9)


def test_compass_azimuth_to_yaw_applies_offset():
    """A non-zero yaw_offset_deg should shift the result by that many degrees."""
    base_yaw = compass_azimuth_to_yaw(70.0, yaw_offset_deg=0.0)
    offset_yaw = compass_azimuth_to_yaw(70.0, yaw_offset_deg=10.0)
    assert offset_yaw == pytest.approx(base_yaw + np.deg2rad(10.0), abs=1e-9)


def test_compass_azimuth_to_yaw_reads_config_when_offset_is_none(monkeypatch):
    """yaw_offset_deg=None must fall back to config.YAW_OFFSET_DEG *at call time*,
    not at import/definition time (see docstring of compass_azimuth_to_yaw)."""
    from ntp_campus import config

    monkeypatch.setattr(config, "YAW_OFFSET_DEG", 15.0)
    yaw_with_config = compass_azimuth_to_yaw(70.0, yaw_offset_deg=None)
    yaw_with_explicit = compass_azimuth_to_yaw(70.0, yaw_offset_deg=15.0)
    assert yaw_with_config == pytest.approx(yaw_with_explicit, abs=1e-9)


# =========================================================================
# SectorAntennaSpec
# =========================================================================
def test_sector_antenna_spec_total_tilt_is_sum_of_mechanical_and_electrical():
    spec = SectorAntennaSpec(
        tx_name="TEST_S1", sector=1, azimuth_deg=0.0,
        electrical_tilt_deg=4.0, mechanical_tilt_deg=2.5,
        height_agl_m=31.8, equipment_notes="test",
    )
    assert spec.total_tilt_deg == pytest.approx(6.5)


def test_sector_antenna_spec_repr_contains_key_fields():
    spec = SectorAntennaSpec(
        tx_name="TEST_S1", sector=1, azimuth_deg=70.0,
        electrical_tilt_deg=4.0, mechanical_tilt_deg=0.0,
        height_agl_m=31.8, equipment_notes="test",
    )
    r = repr(spec)
    assert "TEST_S1" in r
    assert "70.0" in r

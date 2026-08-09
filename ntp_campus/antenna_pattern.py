"""
antenna_pattern.py
===================

Defines the sector specification (SectorAntennaSpec), helper angle conversions,
and – most importantly – the custom "air3278_broadcast" pattern. This pattern
mimics the Ericsson AIR 3278 datasheet (12x1 active array, combined with the
AIR unit into a fixed-beam "broadcast" pattern).

It's registered with Sionna RT via `register_antenna_pattern`, so it can be
used as `pattern="air3278_broadcast"` when creating PlanarArray objects.
"""

import math

import numpy as np
import mitsuba as mi
import drjit as dr
from sionna.rt import PolarizedAntennaPattern, register_antenna_pattern

from . import config


# =========================================================================
# 1. SECTOR SPECIFICATION – metadata per transmitter
# =========================================================================
class SectorAntennaSpec:
    """
    Holds static (planned) parameters for one base-station sector:
    transmitter name, sector number, azimuth, tilt (mechanical + electrical),
    height AGL, and equipment notes.

    `total_tilt_deg` is the sum of mechanical and electrical tilt – that's
    what we actually use when setting the transmitter orientation in Sionna.
    """

    def __init__(self, tx_name, sector, azimuth_deg, electrical_tilt_deg,
                 mechanical_tilt_deg, height_agl_m, equipment_notes):
        self.tx_name = tx_name
        self.sector = sector
        self.azimuth_deg = azimuth_deg
        self.electrical_tilt_deg = electrical_tilt_deg
        self.mechanical_tilt_deg = mechanical_tilt_deg
        self.total_tilt_deg = electrical_tilt_deg + mechanical_tilt_deg
        self.height_agl_m = height_agl_m  # AGL = Above Ground Level
        self.equipment_notes = equipment_notes

    def __repr__(self):
        return (f"SectorAntennaSpec(tx_name={self.tx_name!r}, sector={self.sector}, "
                f"azimuth_deg={self.azimuth_deg}, total_tilt_deg={self.total_tilt_deg})")


# Predefined specs for the three NTP site sectors (AIR 3278, NR3500).
# Actual (x, y) positions are added later in transmitters.py.
AIR3278_SPECS = [
    SectorAntennaSpec("NTP_S1_NR3500", 1, 0.0,   4.0, 0.0, 31.8, "AIR 3278, Radio: 4499B1B3 + 2479B20"),
    SectorAntennaSpec("NTP_S2_NR3500", 2, 70.0,  4.0, 0.0, 31.8, "AIR 3278"),
    SectorAntennaSpec("NTP_S3_NR3500", 3, 140.0, 4.0, 0.0, 31.8, "AIR 3278"),
]

# Same specs, but as dicts with positions – used in radio_map.py for
# site layout drawing and per-sector coverage rendering.
AIR3278_SECTOR_POSITIONS = [
    {"name": "NTP_S1_NR3500", "az": 0.0,   "tilt": 4.0, "h": 31.8, "pos": [-270.63, -152.00]},
    {"name": "NTP_S2_NR3500", "az": 70.0,  "tilt": 4.0, "h": 31.8, "pos": [-270.05, -152.00]},
    {"name": "NTP_S3_NR3500", "az": 140.0, "tilt": 4.0, "h": 31.8, "pos": [-266.56, -159.97]},
]


# =========================================================================
# 2. ANGLE CONVERSION AND HELPERS
# =========================================================================
def compass_azimuth_to_yaw(azimuth_deg, yaw_offset_deg=None):
    """
    Converts a compass azimuth (0° = north, clockwise) to Sionna's yaw angle
    in radians (mathematical convention: 0° = +X, counter-clockwise).

    If yaw_offset_deg is None, uses config.YAW_OFFSET_DEG at call time,
    so changing the config later takes effect immediately – no frozen defaults.
    """
    if yaw_offset_deg is None:
        yaw_offset_deg = config.YAW_OFFSET_DEG
    base = (90.0 - azimuth_deg) % 360.0
    return np.deg2rad(base + yaw_offset_deg)


def to_scalar(x):
    """Safely extracts the first scalar from an arbitrary numpy/drjit/mitsuba array."""
    return float(np.asarray(x).reshape(-1)[0])


# =========================================================================
# 3. CUSTOM PATTERN "air3278_broadcast"
# =========================================================================
def air3278_broadcast_v_pattern(theta: "mi.Float", phi: "mi.Float") -> "mi.Complex2f":
    """
    Analytical (Gaussian) 3D pattern that reproduces the key AIR 3278
    datasheet parameters:
        - V-BW, H-BW (beamwidths)
        - SLL as a floor under the main lobe
        - FBR as a higher floor in the rear hemisphere (|phi| > 90°)
        - peak gain scaled to TARGET_PEAK_GAIN_DBI

    This is not a physically accurate radiation model – it's a deliberately
    simple, parametrised shape that's easy to calibrate against the datasheet.
    (That's what antenna_measurements.py checks.)
    """
    # Fine horizontal asymmetry correction from config.py
    phi_shifted = phi - config.PHI_OFFSET_RAD

    theta_deg = 180.0 * theta / dr.pi
    phi_deg = 180.0 * phi_shifted / dr.pi

    # Gaussian coefficients derived from -3dB beamwidths
    a_v = 2.0 * math.log(2.0) / (config.V_BW_DATASHEET ** 2)
    a_h = 2.0 * math.log(2.0) / (config.H_BW_DATASHEET ** 2)

    # Main lobe: 2D Gaussian, normalised to 1.0 at peak.
    # theta_deg - 90° because theta=90° is the horizon (boresight without tilt).
    main_lobe = dr.exp(-a_v * (theta_deg - 90.0) ** 2 - a_h * phi_deg ** 2)

    # SLL and FBR are given in dB of power, but here we divide by 20 (amplitude)
    # because the gain will be squared later in the path power calculation.
    sll_lin = 10.0 ** (-config.SLL_DATASHEET / 20.0)
    fbr_lin = 10.0 ** (-config.FBR_DATASHEET / 20.0)

    # Side-lobe floor: wherever the main lobe drops below SLL, clamp it.
    gain = dr.maximum(main_lobe, sll_lin)

    # In the rear hemisphere, use a higher floor (fbr_lin) to achieve F/B.
    back_mask = dr.abs(phi_deg) > 90.0
    gain = dr.select(back_mask, dr.maximum(main_lobe, fbr_lin), gain)

    # Final scaling so that the peak equals TARGET_PEAK_GAIN_DBI.
    peak_scale = 10.0 ** (config.TARGET_PEAK_GAIN_DBI / 20.0)
    gain = gain * peak_scale

    return mi.Complex2f(gain, 0.0)


def air3278_broadcast_factory(**kwargs):
    """
    Factory function called by Sionna when creating a PlanarArray with
    pattern="air3278_broadcast". Returns a PolarizedAntennaPattern with
    cross-polarisation (default).
    """
    pol = kwargs.get("polarization", "cross")
    return PolarizedAntennaPattern(v_pattern=air3278_broadcast_v_pattern, polarization=pol)


def register_air3278_pattern():
    """
    Registers the pattern with Sionna RT. Call this once before using it –
    e.g., right after loading the scene in main.py.
    """
    register_antenna_pattern("air3278_broadcast", air3278_broadcast_factory)
    print("Antenna pattern 'air3278_broadcast' registered.")


def apply_antenna_config(antenna_cfg):
    """
    Copies values from run_config.AntennaConfig into the config module.
    air3278_broadcast_v_pattern reads these values on every call, so there's
    no need to re-register – just call this before the next computation.

    Example:
        from ntp_campus.run_config import SimulationConfig
        cfg = SimulationConfig()
        cfg.antenna.h_bw_deg = 60.0
        apply_antenna_config(cfg.antenna)
    """
    config.V_BW_DATASHEET = antenna_cfg.v_bw_deg
    config.H_BW_DATASHEET = antenna_cfg.h_bw_deg
    config.SLL_DATASHEET = antenna_cfg.sll_db
    config.FBR_DATASHEET = antenna_cfg.fbr_db
    config.PHI_OFFSET_DEG = antenna_cfg.phi_offset_deg
    config.PHI_OFFSET_RAD = np.deg2rad(antenna_cfg.phi_offset_deg)
    config.TARGET_PEAK_GAIN_DBI = antenna_cfg.peak_gain_dbi
    config.YAW_OFFSET_DEG = antenna_cfg.yaw_offset_deg
    config.TX_POWER_DBM_DEFAULT = antenna_cfg.tx_power_dbm

    print("Antenna configuration applied: "
          f"V-BW={config.V_BW_DATASHEET}°, H-BW={config.H_BW_DATASHEET}°, "
          f"SLL={config.SLL_DATASHEET}dB, F/B={config.FBR_DATASHEET}dB, "
          f"peak_gain={config.TARGET_PEAK_GAIN_DBI}dBi")

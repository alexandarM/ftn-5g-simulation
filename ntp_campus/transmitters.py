"""
transmitters.py
=================

Places sector transmitters (AIR 3278 base stations, three sectors) into
the campus scene, based on the specifications from `antenna_pattern.py`.

Actual (x, y) coordinates differ slightly per sector (each AIR panel is
physically offset from the mast centre), so they're defined per sector in
`SECTOR_POSITIONS_XY` below.
"""

import numpy as np
from sionna.rt import Transmitter, PlanarArray

from . import config
from .antenna_pattern import AIR3278_SPECS, compass_azimuth_to_yaw


# Actual (x, y) coordinates of each sector on site [m].
SECTOR_POSITIONS_XY = {
    1: (-270.63, -152.00),  # 1: (-270.63, -152.00),
    2: (-270.05, -152.00),  # 2: (-269.70, -150.75),
    3: (-266.56, -159.97),  # 3: (-264.06, -164.85),
}


def set_tx_array(scene, pattern="air3278_broadcast", polarization="cross"):
    """
    Sets `scene.tx_array` (shared by all transmitters in the scene) to a
    1x1 array with the given pattern. Note: the pattern must be registered
    first (see `antenna_pattern.register_air3278_pattern`).
    """
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern=pattern, polarization=polarization,
    )
    return scene.tx_array


def deploy_all_sectors(scene, specs=AIR3278_SPECS, tx_power_dbm=None):
    """
    Removes any existing transmitters and adds all three sectors defined in
    `specs` (default `AIR3278_SPECS`), using coordinates from `SECTOR_POSITIONS_XY`.

    Parameters
    ----------
    scene : sionna.rt.Scene
        The campus scene (with `tx_array` already set).
    specs : list[SectorAntennaSpec]
        Sector specifications (azimuth, tilt, height, etc.).
    tx_power_dbm : float
        Initial per‑transmitter power; can be fine‑tuned later.

    Returns
    -------
    list[sionna.rt.Transmitter]
        List of added transmitters, in the same order as `specs`.
    """
    if tx_power_dbm is None:
        tx_power_dbm = config.TX_POWER_DBM_DEFAULT

    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)

    added = []
    for spec in specs:
        x, y = SECTOR_POSITIONS_XY[spec.sector]
        position = np.array([x, y, spec.height_agl_m], dtype=np.float32)

        yaw = compass_azimuth_to_yaw(spec.azimuth_deg)
        pitch = np.deg2rad(spec.total_tilt_deg)
        orientation = [float(yaw), float(pitch), 0.0]

        tx = Transmitter(name=spec.tx_name, position=position,
                          orientation=orientation, power_dbm=tx_power_dbm)
        scene.add(tx)
        added.append(tx)
        print(f"Added transmitter [{spec.tx_name}] @ sector {spec.sector}, "
              f"azimuth={spec.azimuth_deg}°, tilt={spec.total_tilt_deg}°")

    return added


def deploy_single_sector(scene, sector_position, tx_power_dbm=None):
    """
    Removes all existing transmitters and adds only one sector – used for
    per‑antenna coverage maps (see `radio_map.render_single_sector_coverage`)
    instead of the combined (all three) view.

    Parameters
    ----------
    scene : sionna.rt.Scene
        The campus scene.
    sector_position : dict
        One element from `antenna_pattern.AIR3278_SECTOR_POSITIONS`, e.g.
        {"name": "NTP_S1_NR3500", "az": 0.0, "tilt": 4.0, "h": 31.8,
         "pos": [-270.63, -152.00]}.
    tx_power_dbm : float
        Transmitter power [dBm].

    Returns
    -------
    sionna.rt.Transmitter
        The added transmitter.
    """
    if tx_power_dbm is None:
        tx_power_dbm = config.TX_POWER_DBM_DEFAULT

    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)

    pos = np.array([sector_position["pos"][0], sector_position["pos"][1],
                     sector_position["h"]], dtype=np.float32)
    yaw = compass_azimuth_to_yaw(sector_position["az"])
    pitch = np.deg2rad(sector_position["tilt"])

    tx = Transmitter(
        name=sector_position["name"],
        position=pos,
        orientation=np.array([yaw, pitch, 0.0]),
        power_dbm=tx_power_dbm,
    )
    scene.add(tx)
    print(f"Only antenna in scene: {sector_position['name']} "
          f"(Azimuth: {sector_position['az']}°)")
    return tx


def print_transmitter_powers(scene):
    """Prints current power [dBm] of all transmitters in the scene."""
    for name, tx in scene.transmitters.items():
        p = float(np.asarray(tx.power_dbm).flatten()[0])
        print(f"{name}: power_dbm = {p} dBm")

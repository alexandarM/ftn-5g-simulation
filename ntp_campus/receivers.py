"""
receivers.py
=============

Configures the isotropic RX antenna array and generates random users
(receivers) within the scene boundaries, avoiding positions inside buildings
(checked against 2D bounding boxes of the building meshes).
"""

import random

import numpy as np
from sionna.rt import PlanarArray, Receiver

from . import config


def set_rx_array(scene, pattern="iso", polarization="cross"):
    """
    Sets `scene.rx_array` to an isotropic 1x1 antenna – simulating a typical
    mobile phone that receives equally from all directions.
    """
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern=pattern, polarization=polarization,
    )
    return scene.rx_array


def compute_building_bboxes_2d(groups, ply_folder=config.PLY_FOLDER):
    """
    Computes 2D (x, y) bounding boxes for each building group (output of
    `materials.group_buildings`) by merging vertices from all material parts.
    Used to check whether a point lies inside a building during random user
    generation.

    Returns
    -------
    list[tuple]
        List of (x_min, x_max, y_min, y_max) per building.
    """
    from .materials import _read_ply_vertices
    import os

    bboxes = []
    for name, items in groups.items():
        all_verts = [_read_ply_vertices(os.path.join(ply_folder, item["filename"])) for item in items]
        if all_verts:
            all_verts = np.vstack(all_verts)
            min_xyz = all_verts.min(axis=0)
            max_xyz = all_verts.max(axis=0)
            bboxes.append((min_xyz[0], max_xyz[0], min_xyz[1], max_xyz[1]))
    return bboxes


def generate_random_receiver_positions(building_bboxes_2d,
                                        num_receivers=100,
                                        x_min=config.RX_AREA_X_MIN, x_max=config.RX_AREA_X_MAX,
                                        y_min=config.RX_AREA_Y_MIN, y_max=config.RX_AREA_Y_MAX,
                                        rx_height=config.RX_HEIGHT_M,
                                        max_attempts=20000):
    """
    Generates `num_receivers` random (x, y) positions within a rectangular
    area, rejecting any that fall inside any building bounding box (simple
    rejection sampling).

    Parameters
    ----------
    building_bboxes_2d : list[tuple]
        Output of `compute_building_bboxes_2d`.
    num_receivers : int
        Target number of positions.
    x_min, x_max, y_min, y_max : float
        Sampling area boundaries [m].
    rx_height : float
        Fixed height above ground for all receivers [m].
    max_attempts : int
        Maximum attempts before stopping (prevents infinite loops if building
        density is too high for the requested count).

    Returns
    -------
    list[tuple]
        List of (x, y, z) positions, length <= num_receivers.
    """
    print("\n--- Generating random receivers outside buildings ---")
    print(f"Scene bounds: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")
    print(f"Number of buildings checked: {len(building_bboxes_2d)}")
    print("Generating positions...")

    rx_positions = []
    attempts = 0
    while len(rx_positions) < num_receivers and attempts < max_attempts:
        attempts += 1
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        inside = any(bx_min <= x <= bx_max and by_min <= y <= by_max
                     for (bx_min, bx_max, by_min, by_max) in building_bboxes_2d)

        if not inside:
            rx_positions.append((x, y, rx_height))

    print(f"Generated {len(rx_positions)} receivers after {attempts} attempts.")
    if attempts > 0:
        print(f"Success rate: {len(rx_positions) / attempts * 100:.2f}%")
    return rx_positions


def add_receivers_to_scene(scene, rx_positions, name_prefix="RX"):
    """
    Adds the list of (x, y, z) positions as `Receiver` objects into the scene,
    with names like `RX_001`, `RX_002`, ...

    Returns
    -------
    list[str]
        Names of the added receivers.
    """
    names = []
    for i, (x, y, z) in enumerate(rx_positions):
        rx_name = f"{name_prefix}_{i + 1:03d}"
        rx = Receiver(name=rx_name, position=np.array([x, y, z], dtype=np.float32))
        scene.add(rx)
        names.append(rx_name)

    print(f"Total receivers added: {len(rx_positions)}")
    print("\nFirst 5 receiver positions:")
    for i in range(min(5, len(rx_positions))):
        print(f"  {names[i]}: ({rx_positions[i][0]}, {rx_positions[i][1]}, {rx_positions[i][2]})")

    return names


def deploy_random_users(scene, groups, num_receivers=100, ply_folder=config.PLY_FOLDER):
    """
    High‑level wrapper: computes building bounding boxes, generates random
    positions outside them, and adds them to the scene as receivers.
    .

    Returns
    -------
    list[str]
        Names of the added receivers.
    """
    bboxes = compute_building_bboxes_2d(groups, ply_folder=ply_folder)
    positions = generate_random_receiver_positions(bboxes, num_receivers=num_receivers)
    return add_receivers_to_scene(scene, positions)

"""
radio_map.py
=============

Radio map (coverage) computation using RadioMapSolver, rendering of combined
and per‑sector coverage views, visualisation of antenna site layout, and
sampling/display of user‑to‑sector association based on SINR/path_gain metrics.
"""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sionna.rt import RadioMapSolver, Camera

from .transmitters import deploy_single_sector, set_tx_array


# =========================================================================
# 1. RADIO MAP COMPUTATION (combined – all transmitters simultaneously)
# =========================================================================
def compute_radio_map(scene, max_depth=3, cell_size=(2, 2), samples_per_tx=int(1e6),
                       specular_reflection=True, diffuse_reflection=False,
                       refraction=True, check_diffusion_materials=True):
    """
    Runs RadioMapSolver on the scene and returns the computed radio map
    (path_gain/rss/sinr per cell, per transmitter).

    Parameters
    ----------
    scene : sionna.rt.Scene
        Scene with transmitters (and optionally materials) placed.
    max_depth : int
        Maximum number of interactions per ray path.
    cell_size : tuple(float, float)
        Map cell size [m] in (x, y).
    samples_per_tx : int
        Number of rays emitted per transmitter (resolution/accuracy).
    specular_reflection, diffuse_reflection, refraction : bool
        Which interaction types to include.
    check_diffusion_materials : bool
        If `diffuse_reflection=True`, checks via `materials.ensure_diffusion_ready`
        that scattering materials are assigned; raises a clear error if not.
        Set to False only if you intentionally want to skip the check.

    Returns
    -------
    sionna.rt.RadioMap
    """
    if diffuse_reflection and check_diffusion_materials:
        from .materials import ensure_diffusion_ready
        ensure_diffusion_ready(scene)

    print("\nComputing coverage map (Radio Map)...")
    rm_solver = RadioMapSolver()
    radio_map = rm_solver(
        scene=scene,
        max_depth=max_depth,
        cell_size=list(cell_size),
        samples_per_tx=samples_per_tx,
        specular_reflection=specular_reflection,
        diffuse_reflection=diffuse_reflection,
        refraction=refraction,
    )
    return radio_map


def render_coverage(scene, radio_map, camera=None, rm_tx=None, show_2d_preview=True,
                     resolution=(1024, 1024), num_samples=1024):
    """
    Displays the radio map as a 2D "top‑down" plot (`radio_map.show`) and
    as a 3D scene render with the coverage map overlaid.

    Parameters
    ----------
    scene : sionna.rt.Scene
    radio_map : sionna.rt.RadioMap
        Output of `compute_radio_map`.
    camera : sionna.rt.Camera or None
        Camera for the 3D render; if None, uses a default bird's‑eye view
        above the site centre.
    rm_tx : int or None
        If given, displays the map only for that transmitter index; otherwise
        the combined view of all transmitters.
    show_2d_preview : bool
        Whether to first show a quick 2D `radio_map.show(metric="path_gain")` preview.
    """
    if show_2d_preview:
        print("\nDisplaying radio map preview...")
        try:
            radio_map.show(metric="path_gain")
        except AttributeError:
            pass

    if camera is None:
        camera = Camera(position=[-265, -155, 500], look_at=[-265, -155, 0])

    render_kwargs = dict(
        camera=camera, radio_map=radio_map,
        resolution=list(resolution), num_samples=num_samples,
        rm_show_color_bar=True,
    )
    if rm_tx is not None:
        render_kwargs["rm_tx"] = rm_tx

    scene.render(**render_kwargs)


# =========================================================================
# 2. COVERAGE MAP FOR A SINGLE SECTOR
# =========================================================================
def render_single_sector_coverage(scene, sector_position, tx_power_dbm=None,
                                   camera=None, max_depth=3, cell_size=(2, 2),
                                   samples_per_tx=int(1e6), diffuse_reflection=False):
    """
    Deploys only one sector (`deploy_single_sector`), computes and renders its
    coverage map in isolation. This replaces three almost identical cells from
    the original notebook that differed only by `ACTIVE_INDEX` – now it's a
    single parameter (`sector_position`) and the function is called once per sector.

    Parameters
    ----------
    scene : sionna.rt.Scene
    sector_position : dict
        One element from `antenna_pattern.AIR3278_SECTOR_POSITIONS`.
    tx_power_dbm : float
        Transmitter power [dBm].
    camera : sionna.rt.Camera or None
        Camera for the 3D render (default is a bird's‑eye view over the site).
    diffuse_reflection : bool
        Whether to include diffuse scattering – requires that radio materials
        (`materials.define_radio_materials` + `assign_materials_to_scene`)
        have already been assigned, otherwise `compute_radio_map` raises an error.

    Returns
    -------
    sionna.rt.RadioMap
        The computed radio map for that (single) sector.
    """
    set_tx_array(scene)  # ensure tx_array is set to air3278_broadcast
    deploy_single_sector(scene, sector_position, tx_power_dbm=tx_power_dbm)

    if camera is None:
        camera = Camera(position=[-265, -155, 500], look_at=[-265, -155, 0])

    radio_map = compute_radio_map(scene, max_depth=max_depth, cell_size=cell_size,
                                   samples_per_tx=samples_per_tx,
                                   diffuse_reflection=diffuse_reflection)

    scene.render(camera=camera, radio_map=radio_map, rm_tx=0,
                 resolution=[1024, 1024], num_samples=1024, rm_show_color_bar=True)

    return radio_map


# =========================================================================
# 3. SITE LAYOUT VISUALISATION (antenna positions on the ground)
# =========================================================================
def plot_site_layout(sector_positions, arrow_len=30.0,
                      save_path="figures/site_layout_check_v2.png"):
    """
    Draws a 2D top‑down view of sector positions with arrows indicating
    boresight direction for each antenna. Corresponds to cell 31.

    Parameters
    ----------
    sector_positions : list[dict]
        List in the format of `antenna_pattern.AIR3278_SECTOR_POSITIONS`
        (keys: "name", "az", "pos").
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    xs = [spec["pos"][0] for spec in sector_positions]
    ys = [spec["pos"][1] for spec in sector_positions]

    for spec in sector_positions:
        x0, y0 = spec["pos"]
        az_rad = np.deg2rad(spec["az"])
        # 0° = north = +Y, azimuth increases clockwise
        dx = arrow_len * np.sin(az_rad)
        dy = arrow_len * np.cos(az_rad)

        ax.plot(x0, y0, "ro", markersize=10, zorder=5)
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2.5), zorder=5)
        ax.text(x0 + dx * 1.15, y0 + dy * 1.15,
                f"{spec['name']}\naz={spec['az']}°",
                color="red", fontsize=9, ha="center", zorder=5)

    margin = arrow_len * 1.8
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_xlabel("X (East) [m]")
    ax.set_ylabel("Y (North) [m]")
    ax.set_title("Antenna positions + boresight directions (building footprints omitted)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

    return fig


# =========================================================================
# 4. USER‑TO‑SECTOR ASSOCIATION (sampling positions)
# =========================================================================
def plot_random_user_sampling(radio_map):
    """
    Demonstrates `radio_map.sample_positions` with two different criteria
    (SINR and path_gain), and plots sampled positions over the coverage map.
    Corresponds to cell 37.
    """
    # --- 1. Sampling/visualisation based on SINR metric ---
    pos, cell_ids = radio_map.sample_positions(
        num_pos=100, metric="sinr", min_val_db=20, max_val_db=100,
        min_dist=10, max_dist=400, tx_association=True, center_pos=False,
    )
    fig = radio_map.show(metric="sinr")
    plt.title("Random user positions (SINR, distance, association)")
    for tx, ids in enumerate(cell_ids.numpy()):
        fig.axes[0].plot(ids[:, 1], ids[:, 0], marker="x", linestyle="",
                          color=mpl.colormaps["Paired"].colors[tx])

    # --- 2. Sampling/visualisation based on path_gain metric ---
    pos2, cell_ids2 = radio_map.sample_positions(
        num_pos=100, metric="path_gain", min_val_db=-120,
        min_dist=80, max_dist=300, tx_association=False, center_pos=False,
    )
    fig2 = radio_map.show(metric="path_gain")
    plt.title("Random user positions (Path Gain, distance)")
    for tx, ids in enumerate(cell_ids2.numpy()):
        fig2.axes[0].plot(ids[:, 1], ids[:, 0], marker="x", linestyle="",
                           color=mpl.colormaps["tab10"].colors[tx])

    return {"sinr_positions": pos, "sinr_cell_ids": cell_ids, "sinr_fig": fig,
            "path_gain_positions": pos2, "path_gain_cell_ids": cell_ids2, "path_gain_fig": fig2}


def plot_sector_association(scene, radio_map, num_pos=4, min_val_db=3,
                             min_dist=10, max_dist=200):
    """
    Renders the SINR map from a bird's‑eye view, displays sector association
    (`radio_map.show_association`) and plots a small number of sampled users
    per sector (coloured by sector). Corresponds to cell 39.
    """
    print("Rendering bird's‑eye view with SINR map...")
    cam = Camera(
        position=np.array([0.0, 0.0, 700.0], dtype=np.float32),
        orientation=np.array([0.0, np.pi / 2, -np.pi / 2], dtype=np.float32),
    )
    scene.render(camera=cam, radio_map=radio_map, rm_metric="sinr",
                 rm_vmin=-30, rm_vmax=100, rm_show_color_bar=True)

    print("Displaying sector association...")
    radio_map.show_association("sinr")

    print("Sampling user positions within sectors...")
    pos, cell_ids = radio_map.sample_positions(
        num_pos=num_pos, metric="sinr", min_val_db=min_val_db,
        min_dist=min_dist, max_dist=max_dist, tx_association=True,
    )

    fig = radio_map.show(metric="sinr", vmin=-10)
    plt.title(f"Sampled users by sector (SINR > {min_val_db} dB)")
    for tx, ids in enumerate(cell_ids.numpy()):
        if len(ids) > 0:
            fig.axes[0].plot(ids[:, 1], ids[:, 0], marker="x", linestyle="",
                              color=mpl.colormaps["Dark2"].colors[tx],
                              markersize=10, markeredgewidth=2, label=f"Sector {tx + 1}")
    plt.show()

    return fig, pos, cell_ids


def show_all_radio_map_metrics(radio_map, num_tx=3):
    """
    Displays `path_gain`, `rss`, and `sinr` maps (per transmitter) and their
    cumulative distribution functions (CDFs), both combined and per transmitter.
    Corresponds to cells 41‑46.
    """
    for metric in ("path_gain", "rss"):
        for tx in range(num_tx):
            radio_map.show(metric=metric, tx=tx)

    for metric in ("path_gain", "rss", "sinr"):
        for tx in range(num_tx):
            radio_map.cdf(metric=metric, tx=tx)

    # Combined (all transmitters) CDF curves
    radio_map.cdf(metric="sinr")
    radio_map.cdf(metric="rss")
    radio_map.cdf(metric="path_gain")

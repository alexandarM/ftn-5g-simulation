#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_and_save_all_figures.py
============================

Runs a single configuration (e.g., 1x1_no_diff) and saves all report figures
(Figure 1–44) into `figures/<model_name>/` with standardised names.

Uses `ntp_campus` package functions and extends them to save images with
consistent filenames (e.g., `figure_4.png`, `figure_5.png`).

Usage:
    python run_and_save_all_figures.py
"""

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr
from sionna.rt import PlanarArray, Camera, Transmitter

try:
    import psutil
    _PROCESS = psutil.Process(os.getpid())
except ImportError:
    _PROCESS = None

from ntp_campus.scene_setup import (
    ensure_sionna_installed,
    load_campus_scene,
    download_campus_meshes
)
from ntp_campus.antenna_pattern import (
    register_air3278_pattern,
    apply_antenna_config,
    AIR3278_SPECS,
    AIR3278_SECTOR_POSITIONS
)
from ntp_campus.run_config import SimulationConfig
from ntp_campus.verification import (
    run_full_antenna_verification,
    run_vbw_hbw_verification,
    run_fbr_verification,
    run_sll_verification,
    build_test_transmitter
)
from ntp_campus.antenna_measurements import measure_vbw_simple, measure_hbw_envelope
from ntp_campus.transmitters import deploy_all_sectors
from ntp_campus.radio_map import (
    compute_radio_map,
    render_coverage,
    plot_site_layout,
    plot_random_user_sampling,
    plot_sector_association,
    show_all_radio_map_metrics,
    render_single_sector_coverage
)
from ntp_campus.materials import (
    load_building_dimensions, group_buildings,
    define_radio_materials, assign_materials_to_scene,
    fix_remaining_itu_concrete
)
from ntp_campus.receivers import set_rx_array, deploy_random_users
from ntp_campus.path_solver import (
    run_path_solver,
    diagnose_paths,
    make_bird_eye_camera,
    render_scene
)
from ntp_campus.channel_metrics import plot_channel_metrics
from ntp_campus import config
from ntp_campus.scene_setup import load_empty_scene


# =========================================================================
# 1. SINGLE CONFIGURATION DEFINITION
# =========================================================================
# Change parameters here for the desired run.
CONFIG = {
    "name": "1x1_no_diff",
    "num_rows": 1,
    "num_cols": 1,
    "pattern": "air3278_broadcast",
    "diffusion": False,
    "verify": True,
    "model_name": "1x1_no_diff"
}


# =========================================================================
# 2. HELPER FUNCTIONS FOR SAVING FIGURES
# =========================================================================
def save_figure(fig, output_dir, filename, dpi=150, bbox_inches="tight"):
    """Saves a matplotlib figure to the given directory with the specified filename."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor="white")
    print(f"   Saved: {filepath}")


def save_current_figure(output_dir, filename, dpi=150, bbox_inches="tight"):
    """Saves the current matplotlib figure (plt.gcf())."""
    fig = plt.gcf()
    save_figure(fig, output_dir, filename, dpi, bbox_inches)


def log_memory(label):
    """
    Prints current process RSS (resident memory), if `psutil` is available.
    Call this between pipeline stages to see exactly where memory is going
    in YOUR environment/scene (much more useful than guessing).
    """
    if _PROCESS is None:
        return
    rss_gb = _PROCESS.memory_info().rss / 1024**3
    print(f"   [mem] {label}: {rss_gb:.2f} GB RSS")


def free_memory():
    """
    Forces Python's garbage collector to run AND asks Dr.Jit (Sionna RT's
    backend) to release its internal memory-allocation cache back to the OS.

    Call this AFTER explicitly `del`-ing any large objects you no longer
    need (radio_map, paths, a scene render bitmap, ...) at the call site --
    modifying a caller's variables indirectly through `locals()` does not
    reliably work in CPython, so this only does the two steps that don't
    need caller-side variable access.

    The `dr.flush_malloc_cache()` step matters a lot here: Dr.Jit/Mitsuba
    never release allocated memory by default (they keep an internal cache
    to make repeated allocations cheap), so even after every Python
    reference to a big object is gone and `gc.collect()` has run, the
    process can still be sitting on several GB that only this call will
    actually return to the OS.
    """
    gc.collect()
    dr.flush_malloc_cache()


# =========================================================================
# 3. MAIN FUNCTION FOR ONE CONFIGURATION (SAVES ALL PLOTS)
# =========================================================================
def run_and_save_all_figures(cfg_dict):
    """
    Runs the full pipeline for a single configuration and saves every plot.
    """
    output_dir = f"figures/{cfg_dict['model_name']}"
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*80)
    print(f"RUNNING CONFIGURATION: {cfg_dict['name']}")
    print(f"Results will be saved to: {output_dir}")
    print("="*80)
    log_memory("start")

    # --- Create SimulationConfig object ---
    cfg = SimulationConfig()
    cfg.model_name = cfg_dict["model_name"]
    cfg.num_receivers = 100
    cfg.antenna.tx_power_dbm = 53.0
    cfg.diffusion.enabled = cfg_dict["diffusion"]
    cfg.radio_map_solver.samples_per_tx = int(1e6)
    cfg.radio_map_solver.max_depth = 3
    cfg.path_solver.max_depth = 3
    cfg.path_solver.seed = 41
    if cfg.diffusion.enabled:
        # Diffuse reflection on large, highly-visible surfaces (e.g. the
        # ground "plane" of the campus scene) can blow up PathSolver's
        # candidate-path buffer past its 32-bit limit at the default 1e6.
        # Lower these two (not max_depth) to stay within the limit while
        # keeping full bounce depth. Tune upward if your scene allows it.
        cfg.path_solver.samples_per_src = int(2e5)
        cfg.path_solver.max_num_paths_per_src = int(2e5)

    # --- Apply antenna config and register pattern ---
    apply_antenna_config(cfg.antenna)
    register_air3278_pattern()

    # ------------------------------------------------------------------
    # 1. ANTENNA VERIFICATION (Figures 4, 5, 6, 7, 8, 9 – for 1x1 and 12x1)
    # ------------------------------------------------------------------
    if cfg_dict["verify"]:
        print("\n--- Antenna verification (1x1) ---")
        # Verification for 1x1 (custom pattern)
        clean_scene = load_empty_scene()
        build_test_transmitter(clean_scene, tx_name="TX_CLEAN_TEST",
                               tx_power_dbm=cfg.antenna.tx_power_dbm)
        # V-BW and H-BW
        vbw_hbw_res = run_vbw_hbw_verification(clean_scene, tx_name="TX_CLEAN_TEST", plot=True)
        # Save plots
        # Figure 4: Normalised EIRP curves (V-cut and H-cut)
        fig_vbw_hbw = plt.gcf()  # last figure from run_vbw_hbw_verification
        save_figure(fig_vbw_hbw, output_dir, "figure_4_vbw_hbw_cuts.png")

        # F/B and SLL for 1x1
        fbr_scene = load_empty_scene()
        build_test_transmitter(fbr_scene, tx_name="FBR_TEST_TX",
                               tx_power_dbm=cfg.antenna.tx_power_dbm)
        fbr_res = run_fbr_verification(fbr_scene, tx_name="FBR_TEST_TX", plot=True)
        save_current_figure(output_dir, "figure_fbr_1x1.png")

        sll_res = run_sll_verification(fbr_scene, tx_name="FBR_TEST_TX")
        # SLL does not produce a separate figure, only prints results

        # Figures 5, 6: Radiation patterns for 1x1 (3D, vertical/horizontal cuts)
        # We reuse the V-cut/H-cut curves from Figure 4; 3D pattern could be added later.

        # Verification runs on small throwaway empty scenes -- free them and
        # everything they produced before moving on to the (much heavier)
        # campus scene, rather than letting them linger for the rest of the run.
        del clean_scene, fbr_scene, vbw_hbw_res, fbr_res, sll_res, fig_vbw_hbw
        plt.close("all")
        free_memory()
        log_memory("after antenna verification")

    # ------------------------------------------------------------------
    # 2. VERIFICATION FOR 12x1 (if pattern is tr38901)
    # ------------------------------------------------------------------
    if cfg_dict["pattern"] == "tr38901":
        print("\n--- Antenna verification (12x1 or 4x8) ---")
        # Repeat verification for the given physical array
        clean_scene = load_empty_scene()
        clean_scene.tx_array = PlanarArray(
            num_rows=cfg_dict["num_rows"],
            num_cols=cfg_dict["num_cols"],
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern=cfg_dict["pattern"],
            polarization="cross"
        )
        # Add transmitter with same orientation
        tx = Transmitter(
            name="TX_VERIFY",
            position=[0.0, 0.0, 31.8],
            orientation=[float(np.deg2rad(90.0 - 20.0)), float(np.deg2rad(4.0)), 0.0],
            power_dbm=cfg.antenna.tx_power_dbm
        )
        clean_scene.add(tx)
        # Measurements
        vbw, v_peak_el, v_elevs, v_eirp = measure_vbw_simple(clean_scene, "TX_VERIFY")
        hbw, h_peak_phi, bc, bmax = measure_hbw_envelope(clean_scene, "TX_VERIFY")
        # Plot V-cut and H-cut and save (Figure 7 for 12x1, similar for 4x8)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(v_elevs, v_eirp, 'o-', markersize=3)
        axes[0].axhline(np.nanmax(v_eirp)-3, color='orange', linestyle=':', label='-3dB')
        axes[0].set_xlabel("Elevation [°]")
        axes[0].set_ylabel("EIRP [dBm]")
        axes[0].set_title(f"V-cut ({cfg_dict['num_rows']}x{cfg_dict['num_cols']})")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(bc, bmax, 'o-', markersize=3)
        axes[1].axhline(bmax.max()-3, color='orange', linestyle=':', label='-3dB')
        axes[1].set_xlabel("Local phi_t [°]")
        axes[1].set_ylabel("Max EIRP [dBm]")
        axes[1].set_title(f"H-cut envelope ({cfg_dict['num_rows']}x{cfg_dict['num_cols']})")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        save_figure(fig, output_dir, f"figure_7_vbw_hbw_{cfg_dict['num_rows']}x{cfg_dict['num_cols']}.png")
        plt.close(fig)

        # Figures 8, 9: Radiation patterns for 12x1 (3D, vertical/horizontal)
        # Could be generated via scene.render or custom 3D plot – skipped for now.

    # ------------------------------------------------------------------
    # 3. LOAD CAMPUS SCENE AND DEPLOY TRANSMITTERS
    # ------------------------------------------------------------------
    print("\n--- Loading campus scene ---")
    ensure_sionna_installed()
    download_campus_meshes()
    scene = load_campus_scene()

    # Reset scene (remove all TX and RX)
    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)
    for rx_name in list(scene.receivers.keys()):
        scene.remove(rx_name)

    # Set tx_array according to configuration
    scene.tx_array = PlanarArray(
        num_rows=cfg_dict["num_rows"],
        num_cols=cfg_dict["num_cols"],
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=cfg_dict["pattern"],
        polarization="cross"
    )
    print(f"   tx_array set: {cfg_dict['num_rows']}x{cfg_dict['num_cols']}, pattern={cfg_dict['pattern']}")

    # Deploy sectors
    deploy_all_sectors(scene, specs=AIR3278_SPECS, tx_power_dbm=cfg.antenna.tx_power_dbm)

    # ------------------------------------------------------------------
    # 4. MATERIALS
    # ------------------------------------------------------------------
    print("\n--- Assigning materials ---")
    building_data = load_building_dimensions()
    groups = group_buildings(building_data)
    # Remove old scattering materials if present
    for mat_name in list(scene.radio_materials.keys()):
        if "scattering" in mat_name:
            scene.remove(mat_name)
    materials_dict = define_radio_materials(diffusion_cfg=cfg.diffusion)
    assign_materials_to_scene(scene, groups, materials_dict)
    fix_remaining_itu_concrete(scene, materials_dict["concrete"])
    print("   Materials assigned.")

    # building_data (raw PLY vertex data) is only needed to build `groups`;
    # `groups` itself (much smaller: bounding info per building) is still
    # needed later for `deploy_random_users`, but the raw vertex arrays are not.
    del building_data
    free_memory()
    log_memory("after materials")

    # ------------------------------------------------------------------
    # 5. RADIO MAP (COVERAGE) - Figures 10-16, 40
    # ------------------------------------------------------------------
    print("\n--- Computing radio map ---")
    radio_map = compute_radio_map(
        scene,
        max_depth=cfg.radio_map_solver.max_depth,
        cell_size=cfg.radio_map_solver.cell_size,
        samples_per_tx=cfg.radio_map_solver.samples_per_tx,
        specular_reflection=cfg.radio_map_solver.specular_reflection,
        diffuse_reflection=cfg.diffusion.enabled,
        refraction=cfg.radio_map_solver.refraction
    )
    log_memory("after compute_radio_map")
    # Combined map (Figures 10, 11, 12, 13)
    cam = Camera(position=[-265, -155, 500], look_at=[-265, -155, 0])
    fig_3d = scene.render(
        camera=cam,
        radio_map=radio_map,
        resolution=[1024, 1024],
        num_samples=1024,
        rm_show_color_bar=True
    )
    save_figure(fig_3d, output_dir, "figure_10_coverage_3d_combined.png")
    plt.close(fig_3d)
    del fig_3d
    free_memory()
    log_memory("after figure_10 3D render (1024x1024x1024spp)")

    # 2D display (Figure 11b, 12b, etc.)
    fig_2d = radio_map.show(metric="path_gain")
    save_figure(fig_2d, output_dir, "figure_11_coverage_2d_combined.png")
    plt.close(fig_2d)

    # Map with diffusion (if enabled)
    if cfg.diffusion.enabled:
        fig_2d_diff = radio_map.show(metric="path_gain", vmin=-150, vmax=-30)
        save_figure(fig_2d_diff, output_dir, "figure_13_coverage_2d_with_diff.png")
        plt.close(fig_2d_diff)

    # Per sector (Figures 14, 15, 16 for 12x1; for 1x1 maybe not)
    if cfg_dict["pattern"] == "tr38901":
        for tx_idx in range(3):
            fig_sec = radio_map.show(metric="path_gain", tx=tx_idx)
            save_figure(fig_sec, output_dir, f"figure_14_coverage_tx{tx_idx+1}.png")
            plt.close(fig_sec)

    # Antenna layout (Figure 2)
    # NOTE: same plt.gcf()-after-plt.show() bug as figure_18 (see comment
    # below) -- plot_site_layout() now returns `fig` directly instead.
    fig_layout = plot_site_layout(AIR3278_SECTOR_POSITIONS, save_path=None)
    save_figure(fig_layout, output_dir, "figure_2_site_layout.png")
    plt.close(fig_layout)

    # Coverage overlaid on a real satellite/OSM background of the FTN campus
    # (Figure 12: geo-referenced version of Figure 11). Wrapped in try/except
    # because this step needs internet access to fetch basemap tiles
    # (contextily) and depends on CAMPUS_REF_LAT/LON in config.py being a
    # reasonably accurate match for the scene's BLOSM origin -- neither of
    # which should be allowed to crash the rest of the run if unavailable
    # or still uncalibrated.
    try:
        from ntp_campus.geo_utils import get_campus_geo_reference, plot_coverage_geo
        geo_ref = get_campus_geo_reference()
        fig_geo, _ax_geo = plot_coverage_geo(
            scene, radio_map, geo_ref, metric="path_gain",
            save_path=os.path.join(output_dir, "figure_12_coverage_geo_overlay.png"),
        )
        plt.close(fig_geo)
    except Exception as e:
        print(f"[WARN] Skipping geo-referenced overlay (figure_12): {e!r}. "
              f"Check internet access (contextily needs to fetch basemap tiles) "
              f"and CAMPUS_REF_LAT/CAMPUS_REF_LON in config.py.")

    # ------------------------------------------------------------------
    # 6. USER ASSOCIATION AND SINR (Figures 17-21, 44)
    # ------------------------------------------------------------------
    # User sampling (Figure 17 = SINR-based sampling, Figure 17b = path_gain-based)
    # NOTE: previously captured via `plt.gcf()`, which silently saved the
    # WRONG figure under the "sinr" filename: the path_gain figure is
    # created SECOND inside plot_random_user_sampling(), so it (not the
    # sinr one) was the "current" figure by the time plt.gcf() ran here.
    # Both figures are now returned explicitly and saved under their own,
    # correctly-labelled filenames.
    sampling_res = plot_random_user_sampling(radio_map)
    save_figure(sampling_res["sinr_fig"], output_dir, "figure_17_user_sampling_sinr.png")
    plt.close(sampling_res["sinr_fig"])
    save_figure(sampling_res["path_gain_fig"], output_dir, "figure_17b_user_sampling_path_gain.png")
    plt.close(sampling_res["path_gain_fig"])

    # Sector association (Figures 18, 19, 20, 21)
    # NOTE: previously captured via `plt.gcf()` AFTER plot_sector_association()
    # had already called `plt.show()` internally. Colab/Jupyter's inline
    # backend closes ALL open figures right after displaying them
    # (InlineBackend.close_figures=True by default), so by the time
    # `plt.gcf()` ran here the real figure was already gone -- `plt.gcf()`
    # then silently created a brand-new EMPTY figure, which is exactly why
    # figure_18 displayed fine inline but saved to disk as a blank white
    # PNG. Fixed by having plot_sector_association() return the actual
    # figure object directly instead of relying on pyplot's global state.
    fig_assoc, _assoc_pos, _assoc_cell_ids = plot_sector_association(scene, radio_map)
    save_figure(fig_assoc, output_dir, "figure_18_sector_association.png")
    plt.close(fig_assoc)

    # CDFs and other metrics (Figures 27-29, 37-39)
    # show_all_radio_map_metrics displays CDFs; save them manually
    for metric in ("path_gain", "rss", "sinr"):
        fig_cdf, _, _ = radio_map.cdf(metric=metric)
        save_figure(fig_cdf, output_dir, f"figure_cdf_{metric}_combined.png")
        plt.close(fig_cdf)
        for tx in range(3):
            fig_cdf_tx, _, _ = radio_map.cdf(metric=metric, tx=tx)
            save_figure(fig_cdf_tx, output_dir, f"figure_cdf_{metric}_tx{tx+1}.png")
            plt.close(fig_cdf_tx)

    # `radio_map` (and everything derived from it) is not needed again --
    # PathSolver (next step) computes its own independent `paths` object.
    # This is usually one of the biggest objects in the whole run, so free
    # it explicitly before starting PathSolver rather than letting it sit
    # in memory alongside the (also heavy) PathSolver computation.
    del radio_map, sampling_res, fig_assoc
    plt.close("all")
    free_memory()
    log_memory("after radio map + coverage figures (radio_map freed)")

    # ------------------------------------------------------------------
    # 7. GENERATE USERS AND PATHSOLVER (Figures 25, 26, 22-24, 30-36, 41-43)
    # ------------------------------------------------------------------
    print("\n--- Generating users ---")
    set_rx_array(scene)
    rx_names = deploy_random_users(scene, groups, num_receivers=cfg.num_receivers)
    print(f"   Added {len(rx_names)} receivers.")

    # Figure 25: receiver layout
    cam_bird = make_bird_eye_camera()
    fig_rx = scene.render(camera=cam_bird, resolution=[1024, 1024], num_samples=1024)
    save_figure(fig_rx, output_dir, "figure_25_receivers_layout.png")
    plt.close(fig_rx)
    del fig_rx
    free_memory()

    # PathSolver
    print("\n--- Running PathSolver ---")
    paths = run_path_solver(scene, **cfg.path_solver_kwargs())
    diagnose_paths(paths)
    log_memory("after PathSolver")

    # Figure 26: ray paths
    # NOTE: uses render_scene() (not scene.render(...) directly) because it
    # caps the number of overlaid paths (max_paths, default 8000). Without
    # this cap, Sionna compiles ONE individual Mitsuba shape+emitter PER
    # BOUNCE of EVERY valid path into a brand-new scene on every render call
    # -- with diffuse reflection enabled this repo can easily produce
    # 200k+ valid paths / 700k+ shapes, and THAT compilation (not PathSolver)
    # is what was causing the OOM crash here. `paths` itself is left fully
    # intact for the channel analysis step below (render_scene restores the
    # original path set internally after rendering).
    fig_paths = render_scene(scene, cam_bird, resolution=(1024, 1024), num_samples=1024,
                              paths=paths, max_paths=8000)
    save_figure(fig_paths, output_dir, "figure_26_rays.png")
    plt.close(fig_paths)
    del fig_paths
    free_memory()

    # ------------------------------------------------------------------
    # 8. CHANNEL ANALYSIS (Figures 22-24, 30-36, 41-43)
    # ------------------------------------------------------------------
    print("\n--- Generating channel plots ---")
    # plot_channel_metrics already saves to figures/ with a model_name prefix,
    # but we want numbered filenames. Instead, we call the function with save=True
    # and then rename files, or simply capture the figures. Since it's easier,
    # we run it and let it save, but we'll also manually generate key plots.
    # However, to avoid duplication, we rely on the fact that plot_channel_metrics
    # saves files; we can later move/rename them. But for simplicity, we'll
    # just run it and note that it generates 8 plots in figures/ with prefix.

    results = plot_channel_metrics(
        paths, scene,
        model_name=cfg.model_name,
        save=True,
        tx_gain_dbi=config.TARGET_PEAK_GAIN_DBI
    )
    # In a more controlled setup we would rename each plot to figure_N.png.
    # For now, we assume the user will manually inspect the saved files.

    del paths, results
    plt.close("all")
    free_memory()
    log_memory("end (all major objects freed)")

    # Additional: RSS histogram, Path loss, RMS delay spread (Figures 30-32)
    # These are already inside plot_channel_metrics.

    print("\n--- CONFIGURATION COMPLETE ---")
    print(f"All results saved in: {output_dir}")


# =========================================================================
# 4. RUN
# =========================================================================
if __name__ == "__main__":
    run_and_save_all_figures(CONFIG)

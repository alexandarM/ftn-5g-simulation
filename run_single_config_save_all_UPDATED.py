#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_and_save_all_figures.py
============================

Runs a single configuration (e.g., 1x1_no_diff) and saves all report figures
(Figure 1-44) into `figures/<model_name>/` with standardised names.

FIXED: 
- Removed OSM materials (uses standard suffix-based assignment)
- No return_fig=True (uses plt.gcf() instead)
- Geo overlay matches quick test (metric="rss", alpha=0.5, cmap="plasma")
- Fixed sector association (no TypeError)
- ZIP download at the end
"""

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sionna.rt import PlanarArray, Camera, Transmitter

from ntp_campus.scene_setup import (
    ensure_sionna_installed,
    load_campus_scene,
    download_campus_meshes,
    load_empty_scene
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
from ntp_campus.transmitters import deploy_all_sectors, set_tx_array
from ntp_campus.radio_map import (
    compute_radio_map,
    plot_site_layout,
    plot_random_user_sampling,
    plot_sector_association,
    render_coverage,
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
from ntp_campus.geo_utils import GeoReference, get_campus_geo_reference, plot_coverage_geo

# =========================================================================
# 1. CONFIGURATION
# =========================================================================
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
# 2. HELPER FUNCTIONS
# =========================================================================
def save_figure(fig, output_dir, filename, dpi=150, bbox_inches="tight"):
    """Saves a matplotlib figure and closes it."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor="white")
    print(f"   Saved: {filepath}")
    plt.close(fig)


def save_current_figure(output_dir, filename, dpi=150, bbox_inches="tight"):
    """Saves the current matplotlib figure (plt.gcf()) and closes it."""
    fig = plt.gcf()
    save_figure(fig, output_dir, filename, dpi, bbox_inches)


def zip_figures(config_dict, output_dir="figures", zip_name=None):
    """Zips the figures subfolder (based on model_name) and downloads in Colab."""
    model_name = config_dict["model_name"]
    folder_to_zip = os.path.join(output_dir, model_name)
    if not os.path.exists(folder_to_zip):
        print(f"[ERROR] Folder '{folder_to_zip}' does not exist.")
        return
    if zip_name is None:
        zip_name = f"figures_{model_name}.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_to_zip):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=os.path.dirname(folder_to_zip))
                zipf.write(file_path, arcname)
    print(f"[OK] ZIP created: {zip_name} ({os.path.getsize(zip_name)/1024:.2f} KB)")
    try:
        from google.colab import files
        files.download(zip_name)
        print(f"[OK] Download started: {zip_name}")
    except ImportError:
        print("[INFO] Not in Colab, ZIP saved locally.")


# =========================================================================
# 3. MAIN FUNCTION
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

    # --- Apply antenna config and register pattern ---
    apply_antenna_config(cfg.antenna)
    register_air3278_pattern()

    # ------------------------------------------------------------------
    # 1. ANTENNA VERIFICATION (Figures 4, 5, 6, 7, 8, 9)
    # ------------------------------------------------------------------
    if cfg_dict["verify"]:
        print("\n--- Antenna verification (1x1) ---")
        clean_scene = load_empty_scene()
        build_test_transmitter(clean_scene, tx_name="TX_CLEAN_TEST",
                               tx_power_dbm=cfg.antenna.tx_power_dbm)
        vbw_hbw_res = run_vbw_hbw_verification(clean_scene, tx_name="TX_CLEAN_TEST", plot=False)

        # Figure 4: V-cut and H-cut
        fig_vbw_hbw, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(vbw_hbw_res["v_elevs"], vbw_hbw_res["v_eirp"], 'o-', markersize=3)
        ax1.axhline(np.nanmax(vbw_hbw_res["v_eirp"]) - 3, color='orange', linestyle=':', label='-3dB')
        ax1.set_xlabel("Elevation [°]")
        ax1.set_ylabel("EIRP [dBm]")
        ax1.set_title(f"V-cut (empty scene), V-BW={vbw_hbw_res['vbw']:.2f}°")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(vbw_hbw_res["h_bin_centers"], vbw_hbw_res["h_bin_max_eirp"], 'o-', markersize=3)
        ax2.axhline(np.nanmax(vbw_hbw_res["h_bin_max_eirp"]) - 3, color='orange', linestyle=':', label='-3dB')
        ax2.set_xlabel("Local phi_t [°]")
        ax2.set_ylabel("Max EIRP [dBm]")
        ax2.set_title(f"H-cut envelope (empty scene), H-BW={vbw_hbw_res['hbw']:.2f}°")
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        save_figure(fig_vbw_hbw, output_dir, "figure_4_vbw_hbw_cuts.png")

        # --- F/B ---
        fbr_scene = load_empty_scene()
        build_test_transmitter(fbr_scene, tx_name="FBR_TEST_TX",
                               tx_power_dbm=cfg.antenna.tx_power_dbm)
        fbr_res = run_fbr_verification(fbr_scene, tx_name="FBR_TEST_TX", plot=False)

        fig_fbr, ax = plt.subplots(figsize=(8, 5))
        eirp_back_arr = np.nanmax(fbr_res["eirp_back_grid"], axis=0)
        ax.plot(fbr_res["back_offsets"] - 180.0, eirp_back_arr, "o-",
                color="#E74C3C", label="Rear hemisphere (max over elevation)")
        ax.axhline(fbr_res["eirp_front"], color="#2E86C1", linestyle="--",
                   label=f'Front (boresight) = {float(fbr_res["eirp_front"]):.2f} dBm')
        ax.axhline(fbr_res["eirp_front"] - config.FBR_DATASHEET, color="gray", linestyle=":",
                   label=f"Target: front − {config.FBR_DATASHEET} dB")
        ax.set_xlabel("Offset from 180° [°]")
        ax.set_ylabel("EIRP [dBm]")
        ax.set_title(f'F/B test — measured {float(fbr_res["fbr_measured"]):.2f} dB '
                     f'(target {config.FBR_DATASHEET} dB)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_figure(fig_fbr, output_dir, "figure_fbr_1x1.png")

        # --- SLL ---
        sll_res = run_sll_verification(fbr_scene, tx_name="FBR_TEST_TX")
        fig_sll, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sll_res["elevations"], sll_res["eirp"], 'o-', markersize=3)
        ax.axhline(sll_res["main_peak_db"], color='green', linestyle='--',
                   label=f'Main peak: {float(sll_res["main_peak_db"]):.2f} dBm')
        ax.axhline(sll_res["side_peak_db"], color='red', linestyle='--',
                   label=f'Side peak: {float(sll_res["side_peak_db"]):.2f} dBm')
        ax.set_xlabel("Elevation [°]")
        ax.set_ylabel("EIRP [dBm]")
        ax.set_title(f"Vertical SLL: {float(sll_res['sll_measured']):.2f} dB "
                     f"(target {config.SLL_DATASHEET} dB)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_figure(fig_sll, output_dir, "figure_sll_1x1.png")

    # ------------------------------------------------------------------
    # 2. VERIFICATION FOR 12x1 (if pattern is tr38901)
    # ------------------------------------------------------------------
    if cfg_dict["pattern"] == "tr38901":
        print("\n--- Antenna verification (12x1 or 4x8) ---")
        from ntp_campus.antenna_measurements import measure_vbw_simple, measure_hbw_envelope
        clean_scene = load_empty_scene()
        clean_scene.tx_array = PlanarArray(
            num_rows=cfg_dict["num_rows"],
            num_cols=cfg_dict["num_cols"],
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern=cfg_dict["pattern"],
            polarization="cross"
        )
        tx = Transmitter(
            name="TX_VERIFY",
            position=[0.0, 0.0, 31.8],
            orientation=[float(np.deg2rad(90.0 - 20.0)), float(np.deg2rad(4.0)), 0.0],
            power_dbm=cfg.antenna.tx_power_dbm
        )
        clean_scene.add(tx)
        vbw, v_peak_el, v_elevs, v_eirp = measure_vbw_simple(clean_scene, "TX_VERIFY")
        hbw, h_peak_phi, bc, bmax = measure_hbw_envelope(clean_scene, "TX_VERIFY")

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

    # ------------------------------------------------------------------
    # 3. LOAD CAMPUS SCENE AND DEPLOY TRANSMITTERS
    # ------------------------------------------------------------------
    print("\n--- Loading campus scene ---")
    ensure_sionna_installed()
    download_campus_meshes()
    scene = load_campus_scene()

    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)
    for rx_name in list(scene.receivers.keys()):
        scene.remove(rx_name)

    scene.tx_array = PlanarArray(
        num_rows=cfg_dict["num_rows"],
        num_cols=cfg_dict["num_cols"],
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=cfg_dict["pattern"],
        polarization="cross"
    )
    print(f"   tx_array set: {cfg_dict['num_rows']}x{cfg_dict['num_cols']}, pattern={cfg_dict['pattern']}")

    deploy_all_sectors(scene, specs=AIR3278_SPECS, tx_power_dbm=cfg.antenna.tx_power_dbm)

    # ------------------------------------------------------------------
    # 4. MATERIALS (STANDARD, NO OSM)
    # ------------------------------------------------------------------
    print("\n--- Assigning materials (standard suffix-based) ---")
    building_data = load_building_dimensions()
    groups = group_buildings(building_data)
    # Remove old scattering materials if present
    for mat_name in list(scene.radio_materials.keys()):
        if "scattering" in mat_name:
            scene.remove(mat_name)
    materials_dict = define_radio_materials(diffusion_cfg=cfg.diffusion)
    assign_materials_to_scene(scene, groups, materials_dict)
    fix_remaining_itu_concrete(scene, materials_dict["concrete"])
    print("   Materials assigned (standard).")

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

    # Figure 10: 3D combined coverage
    cam = Camera(position=[-265, -155, 500], look_at=[-265, -155, 0])
    scene.render(
        camera=cam,
        radio_map=radio_map,
        resolution=[1024, 1024],
        num_samples=1024,
        rm_show_color_bar=True
    )
    fig_3d = plt.gcf()
    save_figure(fig_3d, output_dir, "figure_10_coverage_3d_combined.png")

    # Figure 11: 2D combined coverage (Sionna native)
    radio_map.show(metric="path_gain")
    fig_2d = plt.gcf()
    save_figure(fig_2d, output_dir, "figure_11_coverage_2d_combined.png")

    # Figure 11b: GEO overlay (same as quick test)
    print("   Generating geo-referenced coverage overlay...")
    orig_ref = get_campus_geo_reference()
    # Use same rotation/offset as quick test (0,0,0)
    new_ref = GeoReference(
        ref_lat=orig_ref.ref_lat,
        ref_lon=orig_ref.ref_lon,
        rotation_deg=0,
        utm_epsg=orig_ref.utm_epsg
    )
    fig_geo, _ = plot_coverage_geo(
        scene,
        radio_map,
        geo_ref=new_ref,
        metric="rss",          # same as quick test
        alpha=0.5,             # same as quick test
        cmap="plasma",         # same as quick test
        save_path=os.path.join(output_dir, "figure_11b_coverage_geo_overlay.png"),
    )
    plt.close(fig_geo)

    # Figure 13: with diffusion (if enabled)
    if cfg.diffusion.enabled:
        radio_map.show(metric="path_gain", vmin=-150, vmax=-30)
        fig_2d_diff = plt.gcf()
        save_figure(fig_2d_diff, output_dir, "figure_13_coverage_2d_with_diff.png")

    # Per sector (only for tr38901)
    if cfg_dict["pattern"] == "tr38901":
        for tx_idx in range(3):
            radio_map.show(metric="path_gain", tx=tx_idx)
            fig_sec = plt.gcf()
            save_figure(fig_sec, output_dir, f"figure_14_coverage_tx{tx_idx+1}.png")

    # Figure 2: Site layout (save_path works with plot_site_layout)
    plot_site_layout(AIR3278_SECTOR_POSITIONS, save_path=os.path.join(output_dir, "figure_2_site_layout.png"))

    # ------------------------------------------------------------------
    # 6. USER ASSOCIATION AND SINR (Figures 17-21, 44)
    # ------------------------------------------------------------------
    # Figure 17: User sampling
    sampling_res = plot_random_user_sampling(radio_map)
    fig_sampling = plt.gcf()
    save_figure(fig_sampling, output_dir, "figure_17_user_sampling_sinr.png")

    # Figure 18: Sector association (manual, no return_fig)
    print("Generating sector association figure...")
    cam = Camera(position=[0, 0, 700], orientation=[0, np.pi/2, -np.pi/2])
    scene.render(camera=cam, radio_map=radio_map, rm_metric="sinr",
                 rm_vmin=-30, rm_vmax=100, rm_show_color_bar=True)
    fig_assoc = plt.gcf()
    ax = fig_assoc.gca()
    # Add sampled users
    pos, cell_ids = radio_map.sample_positions(
        num_pos=4, metric="sinr", min_val_db=3, min_dist=10, max_dist=200, tx_association=True
    )
    for tx, ids in enumerate(cell_ids.numpy()):
        if len(ids) > 0:
            ax.plot(ids[:, 1], ids[:, 0], marker='x', linestyle='',
                    color=mpl.colormaps['Dark2'].colors[tx],
                    markersize=10, markeredgewidth=2, label=f"Sector {tx+1}")
    ax.set_title("Sector association with sampled users")
    save_figure(fig_assoc, output_dir, "figure_18_sector_association.png")

    # CDFs (Figures 27-29, 37-39)
    for metric in ("path_gain", "rss", "sinr"):
        radio_map.cdf(metric=metric)
        fig_cdf = plt.gcf()
        save_figure(fig_cdf, output_dir, f"figure_cdf_{metric}_combined.png")
        for tx in range(3):
            radio_map.cdf(metric=metric, tx=tx)
            fig_cdf_tx = plt.gcf()
            save_figure(fig_cdf_tx, output_dir, f"figure_cdf_{metric}_tx{tx+1}.png")

    # ------------------------------------------------------------------
    # 7. GENERATE USERS AND PATHSOLVER (Figures 25, 26, 22-24, 30-36, 41-43)
    # ------------------------------------------------------------------
    print("\n--- Generating users ---")
    set_rx_array(scene)
    rx_names = deploy_random_users(scene, groups, num_receivers=cfg.num_receivers)
    print(f"   Added {len(rx_names)} receivers.")

    # Figure 25: receiver layout
    cam_bird = make_bird_eye_camera()
    scene.render(camera=cam_bird, resolution=[1024, 1024], num_samples=1024)
    fig_rx = plt.gcf()
    save_figure(fig_rx, output_dir, "figure_25_receivers_layout.png")

    # PathSolver
    print("\n--- Running PathSolver ---")
    paths = run_path_solver(
        scene,
        max_depth=cfg.path_solver.max_depth,
        los=cfg.path_solver.los,
        specular_reflection=cfg.path_solver.specular_reflection,
        diffuse_reflection=cfg.diffusion.enabled,
        refraction=cfg.path_solver.refraction,
        synthetic_array=cfg.path_solver.synthetic_array,
        seed=cfg.path_solver.seed
    )
    diagnose_paths(paths)

    # Figure 26: ray paths
    scene.render(camera=cam_bird, paths=paths, resolution=[1024, 1024], num_samples=1024)
    fig_paths = plt.gcf()
    save_figure(fig_paths, output_dir, "figure_26_rays.png")

    # ------------------------------------------------------------------
    # 8. CHANNEL ANALYSIS (Figures 22-24, 30-36, 41-43)
    # ------------------------------------------------------------------
    print("\n--- Generating channel plots ---")
    results = plot_channel_metrics(
        paths, scene,
        model_name=cfg.model_name,
        save=True,
        tx_gain_dbi=config.TARGET_PEAK_GAIN_DBI
    )

    print("\n--- CONFIGURATION COMPLETE ---")
    print(f"All results saved in: {output_dir}")

    # ------------------------------------------------------------------
    # 9. ZIP THE FIGURES
    # ------------------------------------------------------------------
    zip_figures(cfg_dict, output_dir="figures", zip_name=f"figures_{cfg_dict['model_name']}.zip")


# =========================================================================
# 4. RUN
# =========================================================================
if __name__ == "__main__":
    run_and_save_all_figures(CONFIG)

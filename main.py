"""
main.py
========

Main entry point for the project – reproduces the full flow of the original
Jupyter notebook as a sequence of clearly named steps that call functions
from the `ntp_campus` package.

CONFIGURATION (antenna pattern, diffusion, RadioMapSolver/PathSolver parameters)
is not modified directly in this file – everything is set via the
`ntp_campus.run_config.SimulationConfig` object below (`SIM_CONFIG`).
For a ready‑to‑paste Colab cell that changes these parameters, see
`colab_config_cell.py` in the repository root.

Running
-------
    python main.py

Note: some steps (e.g., `STEP_DOWNLOAD_MESHES`) assume internet access and
Google Drive availability; if the meshes already exist locally
(`meshes/kampus_test.xml`), that step can be skipped by setting the
corresponding flag to `False` below.
"""

from ntp_campus.run_config import SimulationConfig
from ntp_campus.scene_setup import (
    ensure_sionna_installed, enable_colab_widgets,
    download_campus_meshes, load_campus_scene,
)
from ntp_campus.antenna_pattern import (
    register_air3278_pattern, apply_antenna_config,
    AIR3278_SPECS, AIR3278_SECTOR_POSITIONS,
)
from ntp_campus.verification import run_full_antenna_verification
from ntp_campus.transmitters import set_tx_array, deploy_all_sectors, print_transmitter_powers
from ntp_campus.radio_map import (
    compute_radio_map, render_coverage, render_single_sector_coverage,
    plot_site_layout, plot_random_user_sampling, plot_sector_association,
    show_all_radio_map_metrics,
)
from ntp_campus.materials import (
    load_building_dimensions, group_buildings, print_building_summary,
    print_building_group_details, define_radio_materials,
    assign_materials_to_scene, fix_remaining_itu_concrete,
    print_scene_material_summary, verify_material_assignment,
)
from ntp_campus.receivers import set_rx_array, deploy_random_users
from ntp_campus.path_solver import make_bird_eye_camera, render_scene, run_path_solver, diagnose_paths
from ntp_campus.channel_metrics import plot_channel_metrics
from ntp_campus.export_utils import download_figures_folder


# MAIN CONFIGURATION – CHANGE SIMULATION PARAMETERS HERE
# See ntp_campus/run_config.py for all available fields and explanations.
# Example changes (uncomment / adapt as needed):
#
#   SIM_CONFIG.model_name = "12x1_with_diff"
#   SIM_CONFIG.antenna.h_bw_deg = 60.0
#   SIM_CONFIG.diffusion.enabled = True
#   SIM_CONFIG.radio_map_solver.max_depth = 5
#   SIM_CONFIG.radio_map_solver.samples_per_tx = int(1e7)
#   SIM_CONFIG.path_solver.max_depth = 5
#
SIM_CONFIG = SimulationConfig()


# FLAGS TO ENABLE/DISABLE INDIVIDUAL STEPS
STEP_DOWNLOAD_MESHES = True          # download meshes.zip from Google Drive
STEP_ANTENNA_VERIFICATION = True     # verify V-BW/H-BW/F/B/SLL on an empty scene
STEP_PER_SECTOR_COVERAGE = True      # coverage map for each sector individually
STEP_CHANNEL_ANALYSIS = True         # PathSolver + CIR/CFR/TAPS/RSS analysis
STEP_EXPORT_FIGURES = False          # pack figures/ into ZIP and download (Google Colab only)

# STEP_MATERIALS is not manually turned off when diffusion is enabled – see check below.
STEP_MATERIALS = True
if SIM_CONFIG.diffusion.enabled and not STEP_MATERIALS:
    raise ValueError(
        "SIM_CONFIG.diffusion.enabled=True requires STEP_MATERIALS=True "
        "(diffusion without assigned radio materials has no effect – see "
        "run_config.DiffusionConfig note)."
    )


def main():
    # STEP 1: Environment setup, antenna config, pattern registration

    print(SIM_CONFIG.describe())
    ensure_sionna_installed()
    enable_colab_widgets()
    apply_antenna_config(SIM_CONFIG.antenna)  # before registration, but also works after (read dynamically)
    register_air3278_pattern()


    # STEP 2: Download and load the 3D campus scene

    if STEP_DOWNLOAD_MESHES:
        download_campus_meshes()
    scene = load_campus_scene()


    # STEP 3: Antenna pattern verification on an "empty" scene
    # (compares V-BW/H-BW/F/B/SLL against values from SIM_CONFIG.antenna)

    if STEP_ANTENNA_VERIFICATION:
        run_full_antenna_verification(tx_power_dbm=SIM_CONFIG.antenna.tx_power_dbm)


    # STEP 4: Deploy all three sectors in the campus scene

    set_tx_array(scene)
    deploy_all_sectors(scene, specs=AIR3278_SPECS, tx_power_dbm=SIM_CONFIG.antenna.tx_power_dbm)


    # STEP 5 (part 1): Building geometry and radio material assignment
    # (MUST come before radio map/path computation if diffusion is enabled)

    groups = None
    if STEP_MATERIALS:
        building_data = load_building_dimensions()
        print_building_summary(building_data)
        groups = group_buildings(building_data)
        print_building_group_details(groups, filter_name="NTP")
        print_building_group_details(groups)

        materials = define_radio_materials(diffusion_cfg=SIM_CONFIG.diffusion)
        assign_materials_to_scene(scene, groups, materials)
        fix_remaining_itu_concrete(scene, materials["concrete"])
        print_scene_material_summary(scene)
        verify_material_assignment(scene)


    # STEP 5 (part 2): Combined radio map (all three sectors together)

    radio_map = compute_radio_map(scene, **SIM_CONFIG.radio_map_solver_kwargs())
    render_coverage(scene, radio_map)
    plot_site_layout(AIR3278_SECTOR_POSITIONS)


    # STEP 6: Coverage map for each sector individually

    if STEP_PER_SECTOR_COVERAGE:
        rm_kwargs = SIM_CONFIG.radio_map_solver_kwargs()
        for sector_position in AIR3278_SECTOR_POSITIONS:
            render_single_sector_coverage(
                scene, sector_position,
                tx_power_dbm=SIM_CONFIG.antenna.tx_power_dbm,
                max_depth=rm_kwargs["max_depth"], cell_size=rm_kwargs["cell_size"],
                samples_per_tx=rm_kwargs["samples_per_tx"],
                diffuse_reflection=rm_kwargs["diffuse_reflection"],
            )
        # Restore the scene to all three sectors before continuing the pipeline
        deploy_all_sectors(scene, specs=AIR3278_SPECS, tx_power_dbm=SIM_CONFIG.antenna.tx_power_dbm)
        radio_map = compute_radio_map(scene, **SIM_CONFIG.radio_map_solver_kwargs())


    # STEP 7: User‑to‑sector association (position sampling)

    plot_random_user_sampling(radio_map)
    plot_sector_association(scene, radio_map)
    show_all_radio_map_metrics(radio_map, num_tx=len(AIR3278_SPECS))


    # STEP 8: Generate users (receivers) and compute paths

    if STEP_CHANNEL_ANALYSIS:
        if groups is None:
            building_data = load_building_dimensions()
            groups = group_buildings(building_data)

        set_rx_array(scene)
        deploy_random_users(scene, groups, num_receivers=SIM_CONFIG.num_receivers)

        cam = make_bird_eye_camera()
        render_scene(scene, cam)

        paths = run_path_solver(scene, **SIM_CONFIG.path_solver_kwargs())
        render_scene(scene, cam, paths=paths)
        diagnose_paths(paths)

        print_transmitter_powers(scene)

        results = plot_channel_metrics(paths, scene, model_name=SIM_CONFIG.model_name,
                                        save=True, tx_gain_dbi=SIM_CONFIG.antenna.peak_gain_dbi)
        print("\n[OK] All plots generated!")
        print(f"   Results saved in figures/ folder with prefix: {SIM_CONFIG.model_name}")


    # STEP 9: Export results (Google Colab)

    if STEP_EXPORT_FIGURES:
        download_figures_folder("figures", f"figures_{SIM_CONFIG.model_name}.zip")


if __name__ == "__main__":
    main()

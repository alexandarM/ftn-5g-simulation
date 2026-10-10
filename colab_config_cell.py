"""
colab_config_cell.py
======================

This is NOT part of the importable package – it's a READY‑TO‑USE code snippet
that you can copy directly into a Google Colab cell, right after cloning or
unpacking the repository (so that `ntp_campus/` exists in `/content/`),
and BEFORE you run the `main.py` pipeline.

How to use it in Colab
--------------------------
1. First cell: clone the repo or unzip so that `ntp_campus/` is in `/content/`:

       !git clone https://github.com/<your-account>/<your-repo>.git
       %cd <your-repo>

2. Second cell: COPY the content below (from "BEGIN CELL") and modify values as needed.

3. Third cell: run the pipeline step‑by‑step (see example at the bottom of this file)
   or import and call `main.main()` from `main.py`. However, since `main.py`
   creates its OWN `SIM_CONFIG` at import time, if you want it to use the
   configuration set HERE, the easiest way is to override `main.SIM_CONFIG`
   before calling `main.main()`:

       import main as pipeline
       pipeline.SIM_CONFIG = cfg   # cfg defined in this cell
       pipeline.main()

   Alternatively (recommended for experimentation / quick iteration), call
   functions from `ntp_campus` directly, step by step, as shown in the
   "MANUAL STEP‑BY‑STEP EXECUTION" example at the bottom – that way you have
   full control and see plots immediately after each step, just like in the
   original notebook.
"""

# BEGIN CELL (copy from here into Colab)

from ntp_campus.run_config import SimulationConfig

cfg = SimulationConfig()

# -----------------------------------------------------------------------
# 1) MODEL NAME – used as a suffix for saved plot filenames
#    (e.g., figures/rss_histogram_<model_name>.png, etc.)
# -----------------------------------------------------------------------
cfg.model_name = "12x1_with_diff"   # e.g., "1x1_no_diff", "12x1_with_diff", ...
cfg.num_receivers = 100             # number of random users for channel analysis

# -----------------------------------------------------------------------
# 2) ANTENNA PATTERN (AIR 3278) – change if testing a different antenna
#    or a different calibration of datasheet values.
# -----------------------------------------------------------------------
cfg.antenna.v_bw_deg = 8.5          # vertical beamwidth [°]
cfg.antenna.h_bw_deg = 65.0         # horizontal beamwidth [°]
cfg.antenna.sll_db = 13.0           # side‑lobe level [dB]
cfg.antenna.fbr_db = 25.0           # front‑to‑back ratio [dB]
cfg.antenna.phi_offset_deg = 0.65   # fine horizontal pattern asymmetry correction [°]
cfg.antenna.peak_gain_dbi = 17.0    # peak gain [dBi] (EIRP - P_tx_port)
cfg.antenna.yaw_offset_deg = 0.0    # compass‑to‑Sionna convention calibration [°]
cfg.antenna.tx_power_dbm = 53.0     # power per transmitter port [dBm]

# -----------------------------------------------------------------------
# 3) DIFFUSE SCATTERING (radio materials)
#
#    Note: `cfg.diffusion.enabled = True` only sets `diffuse_reflection=True`
#    in the solvers. For diffusion to actually take effect, the following
#    steps must be called (in main.py this happens automatically in STEP 5,
#    part 1 – BEFORE computing the radio map):
#
#        materials.define_radio_materials(diffusion_cfg=cfg.diffusion)
#        materials.assign_materials_to_scene(scene, groups, materials)
#        materials.fix_remaining_itu_concrete(scene, materials["concrete"])
#
#    If this is not done, `compute_radio_map(...)` and `run_path_solver(...)`
#    will raise a RuntimeError with a clear explanation – no silent ignoring.
# -----------------------------------------------------------------------
cfg.diffusion.enabled = True
cfg.diffusion.concrete_scattering_coefficient = 0.3
cfg.diffusion.brick_scattering_coefficient = 0.4
cfg.diffusion.glass_scattering_coefficient = 0.05
cfg.diffusion.scattering_pattern = "lambertian"

# -----------------------------------------------------------------------
# 4) RADIO MAP SOLVER (statistical coverage map)
# -----------------------------------------------------------------------
cfg.radio_map_solver.max_depth = 5          # 3 without diffusion, 5 typical with diffusion
cfg.radio_map_solver.cell_size = (2.0, 2.0)  # map cell size [m]
cfg.radio_map_solver.samples_per_tx = int(1e7)  # 1e6 fast, 1e7 accurate (slower)
cfg.radio_map_solver.specular_reflection = True
cfg.radio_map_solver.refraction = True
# `diffuse_reflection` is not set here – it follows cfg.diffusion.enabled automatically

# -----------------------------------------------------------------------
# 5) PATH SOLVER (CIR/CFR/TAPS/RSS channel analysis)
# -----------------------------------------------------------------------
cfg.path_solver.max_depth = 5
cfg.path_solver.los = True
cfg.path_solver.specular_reflection = True
cfg.path_solver.refraction = True
cfg.path_solver.synthetic_array = False
cfg.path_solver.seed = 41

# `samples_per_src` / `max_num_paths_per_src`: number of candidate rays
# traced per transmitter during PathSolver's candidate generation (Sionna
# RT default: 1_000_000 each). Its internal candidate-path buffer uses a
# 32-bit index, so enabling diffuse reflection on large, highly-visible
# surfaces (most notably the ground "plane" of the campus scene) can push
# the candidate count past the 2^32 limit and raise an overflow error.
# If that happens, lower THESE two (not max_depth or
# radio_map_solver.samples_per_tx, which are unrelated) — this trims
# candidate-ray density without reducing how many bounces a found path can
# have, so you keep full multi-bounce fidelity for a closer-to-1:1 digital
# twin. A good starting point once diffusion is enabled on large surfaces:
cfg.path_solver.samples_per_src = int(2e5)
cfg.path_solver.max_num_paths_per_src = int(2e5)

# -----------------------------------------------------------------------
# Print the current configuration for a quick check before running
# -----------------------------------------------------------------------
print(cfg.describe())

# =============================================================================
# END CELL
# =============================================================================


# =============================================================================
# MANUAL STEP‑BY‑STEP EXECUTION (optional, instead of main.main())
# =============================================================================
# The following block shows how to use `cfg` with the individual package
# functions, cell by cell in Colab (same flow as main.py but "unrolled" so you
# can see plots after each step). Paste each section into a separate Colab cell.

RUN_MANUAL_EXAMPLE = False  # set to True and run if you want this flow

if RUN_MANUAL_EXAMPLE:
    from ntp_campus.scene_setup import ensure_sionna_installed, enable_colab_widgets, \
        download_campus_meshes, load_campus_scene
    from ntp_campus.antenna_pattern import register_air3278_pattern, apply_antenna_config, \
        AIR3278_SPECS
    from ntp_campus.transmitters import set_tx_array, deploy_all_sectors
    from ntp_campus.materials import (load_building_dimensions, group_buildings,
                                       define_radio_materials, assign_materials_to_scene,
                                       fix_remaining_itu_concrete)
    from ntp_campus.radio_map import compute_radio_map, render_coverage

    # --- Cell A: setup ---
    ensure_sionna_installed()
    enable_colab_widgets()
    apply_antenna_config(cfg.antenna)
    register_air3278_pattern()

    # --- Cell B: scene ---
    download_campus_meshes()
    scene = load_campus_scene()

    # --- Cell C: transmitters ---
    set_tx_array(scene)
    deploy_all_sectors(scene, specs=AIR3278_SPECS, tx_power_dbm=cfg.antenna.tx_power_dbm)

    # --- Cell D: materials (MUST be done before diffusion!) ---
    building_data = load_building_dimensions()
    groups = group_buildings(building_data)
    materials = define_radio_materials(diffusion_cfg=cfg.diffusion)
    assign_materials_to_scene(scene, groups, materials)
    fix_remaining_itu_concrete(scene, materials["concrete"])

    # --- Cell E: radio map (uses all solver parameters from cfg at once) ---
    radio_map = compute_radio_map(scene, **cfg.radio_map_solver_kwargs())
    render_coverage(scene, radio_map)

# NTP Campus 5G Coverage Simulation (Sionna RT)

A 5G network (NR n78, 3.5 GHz) coverage simulation for the NTP campus, utilizing the [Sionna RT](https://nvlabs.github.io/sionna/rt/) ray-tracing library. The project models three sectors of an Ericsson AIR 3278 base station. This includes a custom antenna pattern calibrated to datasheet values (V-BW, H-BW, F/B, SLL), radio materials for the campus buildings (concrete/brick/glass with diffuse scattering), and comprehensive channel analysis (CIR/CFR/TAPS/RSS/delay spread).

This repository was created by refactoring the original research Jupyter notebook into a modular Python package (`ntp_campus/`) to improve readability, simplify testing, and facilitate further development.

## Project Structure

```text
sionna_coverage/
├── main.py                        # Main pipeline (replaces the notebook)
├── run_single_config_save_all.py  # Runs one config end-to-end and saves all report figures (Figure 1-44) with standardised names
├── colab_config_cell.py           # Ready-to-use Colab cell for setting up SimulationConfig
├── requirements.txt
├── ntp_campus/
│   ├── __init__.py
│   ├── config.py                  # All constants and adjustable parameters
│   ├── run_config.py              # SimulationConfig - central configuration (antenna/diffusion/solvers)
│   ├── antenna_pattern.py         # Custom AIR3278 antenna pattern + registration
│   ├── antenna_measurements.py    # V-BW / H-BW / F/B / SLL measurements
│   ├── verification.py            # Verification orchestration on an empty scene
│   ├── scene_setup.py             # Downloading and loading the 3D campus scene
│   ├── transmitters.py            # Sector transmitter placement
│   ├── materials.py               # PLY building geometry + radio material assignment
│   ├── receivers.py               # Random user (RX) generation
│   ├── radio_map.py               # Radio map calculation and visualization (coverage)
│   ├── path_solver.py             # PathSolver trajectory calculation and diagnostics
│   ├── channel_metrics.py         # CIR/CFR/TAPS/RSS analysis and plotting
│   ├── geo_utils.py               # WGS84 georeferencing + satellite/OSM overlay of coverage maps
│   └── export_utils.py            # Zipping and downloading the figures/ folder (Google Colab)
└── figures/                       # All generated plots (created at runtime)
```

## Configuration Setup (Antenna, Diffusion, Solvers)

All parameters that a user might typically want to modify from a Google Colab cell (such as the antenna pattern, enabling diffuse scattering, or tweaking `RadioMapSolver` and `PathSolver` settings) are grouped into a single object: `ntp_campus.run_config.SimulationConfig`. The `main.py` script creates this object as `SIM_CONFIG` at the top of the file, which is where you can adjust settings for local execution.

If you are using Colab, check out **`colab_config_cell.py`**. It contains ready-to-use code that you can copy and paste into a single cell, covering:

```python
from ntp_campus.run_config import SimulationConfig

cfg = SimulationConfig()
cfg.model_name = "12x1_with_diff"

cfg.antenna.h_bw_deg = 65.0          # Change the antenna pattern
cfg.antenna.tx_power_dbm = 53.0

cfg.diffusion.enabled = True         # Enable diffuse scattering
cfg.diffusion.concrete_scattering_coefficient = 0.3

cfg.radio_map_solver.max_depth = 5
cfg.radio_map_solver.samples_per_tx = int(1e7)

cfg.path_solver.max_depth = 5
cfg.path_solver.seed = 41
```

**Note: Diffusion requires materials.** Setting `cfg.diffusion.enabled = True` only enables `diffuse_reflection=True` within the solvers. This will have no effect until the objects in the scene are assigned radio materials with a scattering coefficient. This assignment process happens via `materials.define_radio_materials()` -> `materials.assign_materials_to_scene()` -> `materials.fix_remaining_itu_concrete()`, all of which occur in step 5 of `main.py`, BEFORE the radio map calculation. If you try to run the solver with diffusion enabled before this setup is complete, `compute_radio_map()` or `run_path_solver()` will immediately raise a clear `RuntimeError` rather than silently returning results identical to a non-diffuse simulation. You can check `materials.ensure_diffusion_ready()` for more details.

## Running the Simulation

The project was originally developed for **Google Colab** to take advantage of GPU support and fetch meshes from Google Drive via `gdown`. However, every function takes `scene` as an explicit parameter, meaning the entire package can be run locally or on a dedicated server.

```bash
pip install -r requirements.txt
python main.py
```

The execution flags at the top of `main.py` (`STEP_DOWNLOAD_MESHES`, `STEP_ANTENNA_VERIFICATION`, `STEP_PER_SECTOR_COVERAGE`, `STEP_MATERIALS`, `STEP_CHANNEL_ANALYSIS`, `STEP_EXPORT_FIGURES`) allow you to easily toggle specific pipeline steps. For example, if you have already downloaded the meshes locally, you can set `STEP_DOWNLOAD_MESHES = False`.

### Generating all report figures (`run_single_config_save_all.py`)

`main.py` is the general-purpose pipeline for exploring/debugging a configuration interactively. `run_single_config_save_all.py` is a separate, report-oriented entry point built on top of the same `ntp_campus` functions: it runs one `SimulationConfig` end-to-end and saves every figure used in the project report (Figure 1-44) into `figures/<model_name>/` with standardised, numbered filenames (`figure_4.png`, `figure_5.png`, ...), instead of the free-form names produced by `main.py`/`channel_metrics.py`. It also logs memory usage between steps (via `psutil`, if installed) and calls `geo_utils.plot_coverage_geo()` automatically (Figure 12), skipping it gracefully if `contextily` can't reach the internet for basemap tiles.

Use `main.py` for iterating on a configuration; use `run_single_config_save_all.py` when you want the full, consistently-named figure set for the report:

```bash
python run_single_config_save_all.py
```

## Simulation Flow (Steps in `main.py`)

1. **Environment Preparation** - Installs Sionna RT (if missing) and registers the custom `air3278_broadcast` antenna pattern.
2. **Scene Loading** - Downloads the 3D campus meshes, loads them into Sionna RT, and sets the operating frequency (3.5 GHz).
3. **Antenna Verification** - Uses a completely empty scene (without buildings) to measure the vertical/horizontal beamwidth, front-to-back ratio, and side-lobe level, comparing them against the datasheet values defined in `config.py`.
4. **Sector Placement** - Adds three AIR 3278 sectors to the campus scene at their real-world positions, azimuths, and tilts.
5. **Radio Map (Coverage)** - Calculates and renders the combined coverage for all three sectors.
6. **Per-Sector Coverage** - Generates isolated radio maps for each individual sector.
7. **User Association** - Samples user positions based on SINR/path_gain metrics and associates them with the best-serving sector.
8. **Radio Materials** - Reads building geometry from PLY files and assigns materials (concrete/brick/glass) along with their diffuse scattering parameters.
9. **Channel Analysis** - Generates 100 random outdoor users, computes paths using the `PathSolver`, and performs a comprehensive channel analysis (CIR, CFR, TAPS, RSS, path loss, RMS delay spread, interaction distribution). All resulting plots are saved in the `figures/` directory.
10. **Export** - (Optional) Packages the `figures/` directory into a ZIP file for easy download in Google Colab.

## Notes on the Antenna Pattern

The `ntp_campus.antenna_pattern.air3278_broadcast_v_pattern` is an analytical (Gaussian) model. It is NOT a physically exact reproduction of the actual AIR 3278 panel radiation. Instead, it is a deliberately simplified shape parameterized directly by datasheet values (V-BW, H-BW, SLL, F/B, peak gain). The functions provided in `ntp_campus.antenna_measurements` are there specifically to verify that the simulated pattern meets these target values before it is deployed for coverage calculations across the full campus scene.

## Dependencies

Please refer to `requirements.txt`. Key libraries include: `sionna-rt`, `mitsuba`, `drjit`, `numpy`, `matplotlib`, `plyfile`, `shapely`, `ipywidgets`, `psutil` (memory logging in `run_single_config_save_all.py`), and `pyproj` / `contextily` (georeferencing, see below).

## Georeferencing (`geo_utils.py`)

`ntp_campus/geo_utils.py` maps between the Sionna scene's local (x, y) coordinates and real WGS84 lat/lon, so coverage maps can be overlaid on a real satellite/OSM background of the FTN campus (similar to the BostonTwin project).

```python
from ntp_campus.geo_utils import get_campus_geo_reference, plot_coverage_geo

geo_ref = get_campus_geo_reference()
plot_coverage_geo(scene, radio_map, geo_ref, metric="path_gain",
                   save_path="figures/coverage_geo_overlay.png")
```

The reference point is set in `config.py` (`CAMPUS_REF_LAT`, `CAMPUS_REF_LON`, `CAMPUS_REF_ROTATION_DEG`), defaulted to the FTN campus address. This is only a starting approximation — the value that lines up exactly with the imported geometry depends on wherever (0, 0) was placed in Blender during the BLOSM import. Refine it by checking whether the transmitter markers in `plot_coverage_geo()`'s output land on the real mast locations on the satellite image; adjust `CAMPUS_REF_LAT`/`CAMPUS_REF_LON` (and `CAMPUS_REF_ROTATION_DEG` if the import isn't aligned with true north) until they do.

`run_single_config_save_all.py` calls this automatically (Figure 12) and skips it gracefully (with a warning, not a crash) if `contextily` can't reach the internet to fetch basemap tiles.
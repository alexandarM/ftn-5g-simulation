"""
run_config.py
================

Centralised, easy-to-tweak configuration for a single simulation run.

While `config.py` holds "factory" package settings (datasheet values, disk paths),
`run_config.py` is meant for the USER – dataclasses instantiated and modified
at the top of the Colab notebook / script, BEFORE calling `main()` or individual
pipeline steps.

Grouped into four logical blocks:
    AntennaConfig          – pattern parameters (V-BW, H-BW, SLL, F/B, peak gain,
                              yaw offset, TX power)
    DiffusionConfig        – whether diffuse scattering is enabled and with which
                              coefficients for concrete/brick/glass
    RadioMapSolverConfig   – RadioMapSolver parameters (max_depth, cell_size,
                              samples_per_tx, specular/refraction)
    PathSolverConfig       – PathSolver parameters (max_depth, los,
                              specular/refraction, synthetic_array, seed)

All four are bundled into `SimulationConfig`, which has helper methods
`radio_map_solver_kwargs()` and `path_solver_kwargs()` that return `**kwargs`
ready to pass to `radio_map.compute_radio_map(...)` / `path_solver.run_path_solver(...)`.

IMPORTANT NOTE ABOUT DIFFUSION
------------------------------
`diffuse_reflection=True` in the solvers has NO EFFECT if scene objects still use
the default built‑in radio material from the XML, because that material has
`scattering_coefficient=0`. Diffusion requires that
`materials.define_radio_materials()` + `materials.assign_materials_to_scene()`
(+ `materials.fix_remaining_itu_concrete()`) have been called on the scene BEFORE
running the solver with `diffusion.enabled=True`. Functions in `radio_map.py` and
`path_solver.py` automatically check this (via `materials.ensure_diffusion_ready`)
and raise a clear error if materials aren't ready, instead of silently returning
wrong (identical to "no‑diffusion") results.
"""

from dataclasses import dataclass, field
from typing import Tuple


# =========================================================================
# 1. ANTENNA PATTERN
# =========================================================================
@dataclass
class AntennaConfig:
    """
    Parameters defining the shape of the custom "air3278_broadcast" pattern
    (see `antenna_pattern.py`) and transmitter power. Change these and call
    `antenna_pattern.apply_antenna_config(cfg.antenna)` BEFORE registering the
    pattern (or again before the next solver call – changes take effect
    immediately because the pattern reads from the `config` module on each call,
    not at registration time).
    """
    v_bw_deg: float = 8.5          # Vertical -3 dB beamwidth [degrees]
    h_bw_deg: float = 65.0         # Horizontal -3 dB beamwidth [degrees]
    sll_db: float = 13.0           # Side-lobe level [dB]
    fbr_db: float = 25.0           # Front-to-back ratio [dB]
    phi_offset_deg: float = 0.65   # Fine horizontal asymmetry correction [degrees]
    peak_gain_dbi: float = 17.0    # Peak gain (EIRP_max - P_tx_port) [dBi]
    yaw_offset_deg: float = 0.0    # Compass -> Sionna convention calibration [degrees]
    tx_power_dbm: float = 53.0     # Power per TX port [dBm]


# =========================================================================
# 2. DIFFUSE SCATTERING (RADIO MATERIALS)
# =========================================================================
@dataclass
class DiffusionConfig:
    """
    Enables/disables diffuse scattering and sets the scattering coefficients
    for materials used in `materials.define_radio_materials()`.

    `enabled=False` (default) -> solvers are called with `diffuse_reflection=False`;
    materials can still be assigned (no harm), but their scattering coefficients
    are ignored.

    `enabled=True` -> BEFORE running the solver, you MUST call
    `materials.define_radio_materials()` + `materials.assign_materials_to_scene()`
    on the scene (see note in the module docstring above).
    """
    enabled: bool = False
    concrete_scattering_coefficient: float = 0.3
    brick_scattering_coefficient: float = 0.4
    glass_scattering_coefficient: float = 0.05
    scattering_pattern: str = "lambertian"


# =========================================================================
# 3. RADIO MAP SOLVER (statistical coverage map)
# =========================================================================
@dataclass
class RadioMapSolverConfig:
    """
    Parameters for the RadioMapSolver call. `diffuse_reflection` is NOT set
    directly here – it's derived from `DiffusionConfig.enabled` via
    `SimulationConfig.radio_map_solver_kwargs()`, to avoid inconsistencies.
    """
    max_depth: int = 3                     # 3 = without diffusion (faster); 5 = typical with diffusion
    cell_size: Tuple[float, float] = (2.0, 2.0)  # Map cell size [m]
    samples_per_tx: int = int(1e6)         # 1e6 = fast/exploratory; 1e7 = accurate (slower)
    specular_reflection: bool = True
    refraction: bool = True


# =========================================================================
# 4. PATH SOLVER (actual paths / CIR / CFR / RSS analysis)
# =========================================================================
@dataclass
class PathSolverConfig:
    """
    Parameters for the PathSolver call. As with RadioMapSolverConfig,
    `diffuse_reflection` is derived from `DiffusionConfig.enabled`.

    NOTE ON `samples_per_src` / `max_num_paths_per_src`
    -----------------------------------------------------
    These control the number of candidate rays traced per transmitter
    (and the candidate-path buffer size) during PathSolver's candidate
    generation. Sionna RT defaults both to 1_000_000. Its internal
    candidate-path buffer uses a 32-bit index, so enabling
    `diffuse_reflection` on large, highly-visible surfaces (most notably
    the ground "plane" of a campus/street scene) can push the number of
    candidate diffuse interactions per source past the 2^32 limit,
    raising an overflow error. If you hit that, lower THESE two values
    (not `max_depth` or `RadioMapSolverConfig.samples_per_tx`, which are
    unrelated) — this reduces candidate-ray density without reducing how
    many bounces a found path can have, so you can keep a higher
    `max_depth` for closer-to-1:1 digital-twin fidelity. Start around
    1e5-5e5 when diffuse reflection is enabled on large surfaces, and
    tune upward from there until you hit the limit again or compute time
    becomes prohibitive.
    """
    max_depth: int = 3            # 3 = without diffusion; 5 = typical with diffusion
    los: bool = True
    specular_reflection: bool = True
    refraction: bool = True
    synthetic_array: bool = False
    seed: int = 41                # Random seed for reproducibility (e.g., diffuse scattering)
    samples_per_src: int = int(1e6)        # candidate rays per TX (lower if diffuse + large surfaces -> 2^32 overflow)
    max_num_paths_per_src: int = int(1e6)  # candidate-path buffer size per TX (lower alongside samples_per_src)


# =========================================================================
# 5. COMBINED CONFIGURATION FOR A SINGLE RUN
# =========================================================================
@dataclass
class SimulationConfig:
    """
    All four parameter groups plus general run settings (model name – used as
    a suffix in saved plot filenames – and number of random users for channel
    analysis).

    Example usage (Colab cell) – see also `colab_config_cell.py` in the repo
    root for a full copy‑paste example:

        from ntp_campus.run_config import SimulationConfig

        cfg = SimulationConfig()
        cfg.model_name = "12x1_with_diff"
        cfg.antenna.h_bw_deg = 60.0          # change H-BW
        cfg.diffusion.enabled = True          # enable diffusion
        cfg.radio_map_solver.max_depth = 5
        cfg.radio_map_solver.samples_per_tx = int(1e7)
        cfg.path_solver.max_depth = 5
        cfg.path_solver.seed = 123
    """
    model_name: str = "1x1_no_diff"
    num_receivers: int = 100

    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    radio_map_solver: RadioMapSolverConfig = field(default_factory=RadioMapSolverConfig)
    path_solver: PathSolverConfig = field(default_factory=PathSolverConfig)

    def radio_map_solver_kwargs(self) -> dict:
        """Returns kwargs ready for `radio_map.compute_radio_map(scene, **kwargs)`."""
        return dict(
            max_depth=self.radio_map_solver.max_depth,
            cell_size=tuple(self.radio_map_solver.cell_size),
            samples_per_tx=self.radio_map_solver.samples_per_tx,
            specular_reflection=self.radio_map_solver.specular_reflection,
            diffuse_reflection=self.diffusion.enabled,
            refraction=self.radio_map_solver.refraction,
        )

    def path_solver_kwargs(self) -> dict:
        """Returns kwargs ready for `path_solver.run_path_solver(scene, **kwargs)`."""
        return dict(
            max_depth=self.path_solver.max_depth,
            los=self.path_solver.los,
            specular_reflection=self.path_solver.specular_reflection,
            diffuse_reflection=self.diffusion.enabled,
            refraction=self.path_solver.refraction,
            synthetic_array=self.path_solver.synthetic_array,
            seed=self.path_solver.seed,
            samples_per_src=self.path_solver.samples_per_src,
            max_num_paths_per_src=self.path_solver.max_num_paths_per_src,
        )

    def describe(self) -> str:
        """Short readable dump of the current config – useful for logging before running."""
        d = self.diffusion
        lines = [
            f"SimulationConfig(model_name={self.model_name!r}, num_receivers={self.num_receivers})",
            f"  Antenna: V-BW={self.antenna.v_bw_deg}°, H-BW={self.antenna.h_bw_deg}°, "
            f"SLL={self.antenna.sll_db}dB, F/B={self.antenna.fbr_db}dB, "
            f"peak_gain={self.antenna.peak_gain_dbi}dBi, tx_power={self.antenna.tx_power_dbm}dBm",
            f"  Diffusion: {'ENABLED' if d.enabled else 'disabled'}"
            + (f" (concrete={d.concrete_scattering_coefficient}, brick={d.brick_scattering_coefficient}, "
               f"glass={d.glass_scattering_coefficient}, pattern={d.scattering_pattern!r})" if d.enabled else ""),
            f"  RadioMapSolver: max_depth={self.radio_map_solver.max_depth}, "
            f"cell_size={self.radio_map_solver.cell_size}, "
            f"samples_per_tx={self.radio_map_solver.samples_per_tx:.0e}, "
            f"diffuse_reflection={d.enabled}",
            f"  PathSolver: max_depth={self.path_solver.max_depth}, "
            f"seed={self.path_solver.seed}, diffuse_reflection={d.enabled}, "
            f"synthetic_array={self.path_solver.synthetic_array}, "
            f"samples_per_src={self.path_solver.samples_per_src:.0e}, "
            f"max_num_paths_per_src={self.path_solver.max_num_paths_per_src:.0e}",
        ]
        return "\n".join(lines)

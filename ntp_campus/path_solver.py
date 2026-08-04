"""
path_solver.py
================

Runs PathSolver (ray-tracing for actual paths, as opposed to the statistical
RadioMapSolver), renders paths over the scene, and provides diagnostic output
(total valid paths, interaction type distribution, LoS/NLoS statistics) to
verify that the computation makes sense before moving on to channel analysis
(channel_metrics.py).
"""

import numpy as np
from sionna.rt import PathSolver, Camera
from sionna.rt.constants import InteractionType


def make_bird_eye_camera(position=(200.0, -155.0, 800.0), look_at=(0.0, 0.0, 0.0)):
    """Creates a bird's‑eye camera above the campus for scene/path previews."""
    return Camera(
        position=np.array(position, dtype=np.float32),
        look_at=np.array(look_at, dtype=np.float32),
    )


def render_scene(scene, camera, resolution=(1024, 1024), num_samples=1024, paths=None):
    """
    Renders a 3D view of the scene, optionally with `paths` overlaid.
    Corresponds to cells 66 and 70.
    """
    kwargs = dict(camera=camera, resolution=list(resolution), num_samples=num_samples)
    if paths is not None:
        kwargs["paths"] = paths
    scene.render(**kwargs)


def run_path_solver(scene, max_depth=3, los=True, specular_reflection=True,
                     diffuse_reflection=False, refraction=True,
                     synthetic_array=False, seed=41, check_diffusion_materials=True):
    """
    Runs PathSolver on the scene and returns the `paths` object containing
    full channel information (complex gains, delays, angles, interaction types)
    for all (rx, tx) pairs.

    Parameters
    ----------
    scene : sionna.rt.Scene
        Scene with transmitters and receivers placed.
    max_depth : int
        Maximum number of interactions per path.
    los : bool
        Whether to include Line-of-Sight.
    specular_reflection, diffuse_reflection, refraction : bool
        Which interaction types to include.
    synthetic_array : bool
        If True, uses the faster analytical array model instead of explicit
        per‑element ray‑tracing.
    seed : int
        Random seed for reproducibility (e.g., diffuse scattering).
    check_diffusion_materials : bool
        If `diffuse_reflection=True`, automatically checks that scattering
        materials are assigned (see `radio_map.compute_radio_map`).

    Returns
    -------
    sionna.rt.Paths
    """
    if diffuse_reflection and check_diffusion_materials:
        from .materials import ensure_diffusion_ready
        ensure_diffusion_ready(scene)

    p_solver = PathSolver()
    paths = p_solver(
        scene=scene, max_depth=max_depth, los=los,
        specular_reflection=specular_reflection,
        diffuse_reflection=diffuse_reflection,
        refraction=refraction, synthetic_array=synthetic_array, seed=seed,
    )
    return paths


def diagnose_paths(paths):
    """
    Detailed diagnostic printout of the `paths` object: array shapes (a, tau,
    valid, interactions), total valid paths, statistics per (rx, tx) pair,
    interaction type distribution (NONE/SPECULAR/DIFFUSE/REFRACTION/DIFFRACTION),
    number of pure LoS paths, and example data for the first (rx, tx) pair.
    Corresponds to cell 72 – useful for a quick sanity check before channel analysis.

    Returns
    -------
    dict
        Summary of key numbers (for programmatic use):
        total_valid_paths, valid_per_pair (np.ndarray), interaction_counts (dict).
    """
    print("=" * 70)
    print("PATHS OBJECT DIAGNOSTICS (Sionna 2.x)")
    print("=" * 70)

    # --- a (complex gain) is a Tuple(real, imag) ---
    a_real, a_imag = paths.a
    a_real_np = np.array(a_real)
    a_imag_np = np.array(a_imag)
    a_complex = a_real_np + 1j * a_imag_np

    print(f"\nShape of a_real / a_imag: {a_real_np.shape}")
    print("  -> [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]")

    tau_np = np.array(paths.tau)
    print(f"\nShape of tau: {tau_np.shape}")

    valid_np = np.array(paths.valid).astype(bool)
    print(f"\nShape of valid: {valid_np.shape}")
    print(f"Total VALID paths (all rx-tx-antenna pairs): {valid_np.sum()}")
    print(f"Maximum possible path slots: {valid_np.size}")

    valid_per_pair = valid_np.sum(axis=-1)
    print(f"\nStatistics of valid path count per (rx,rx_ant,tx,tx_ant) pair:")
    print(f"  min: {valid_per_pair.min()}, max: {valid_per_pair.max()}, "
          f"mean: {valid_per_pair.mean()}")

    # --- interactions ---
    inter_np = np.array(paths.interactions).astype(np.uint32)
    print(f"\nShape of interactions: {inter_np.shape}")
    print("  -> [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]")

    print("\nInteraction type distribution (all values, including NONE for "
          "empty slots/shorter paths):")
    unique_vals, counts = np.unique(inter_np, return_counts=True)
    type_names = {0: "NONE", 1: "SPECULAR", 2: "DIFFUSE", 4: "REFRACTION", 8: "DIFFRACTION"}
    interaction_counts = {}
    for v, c in zip(unique_vals, counts):
        name = type_names.get(int(v), f"unknown({v})")
        interaction_counts[name] = int(c)
        print(f"  {name} ({v}): {c}")

    has_specular = np.any(inter_np == InteractionType.SPECULAR, axis=0)
    print(f"\nNumber of paths with at least one specular reflection: {has_specular.sum()}")

    has_diffuse = np.any(inter_np == InteractionType.DIFFUSE, axis=0)
    print(f"Number of paths with at least one diffuse reflection: {has_diffuse.sum()}")

    has_refraction = np.any(inter_np == InteractionType.REFRACTION, axis=0)
    print(f"Number of paths with at least one refraction: {has_refraction.sum()}")

    is_los_only = np.all(inter_np == InteractionType.NONE, axis=0) & valid_np
    print(f"Number of pure LoS paths (valid, no interactions): {is_los_only.sum()}")

    if valid_np.any():
        print(f"\ntau (only valid paths) - min: {tau_np[valid_np].min()}, "
              f"max: {tau_np[valid_np].max()}")
        print(f"Number of UNIQUE tau values (valid paths): "
              f"{len(np.unique(tau_np[valid_np]))}")
    else:
        print("\ntau (only valid paths) - min: N/A, max: N/A")
        print("Number of UNIQUE tau values (valid paths): 0")

    print(f"\n--- Example: rx=0, rx_ant=0, tx=0, tx_ant=0 ---")
    print(f"tau: {tau_np[0, 0, 0, 0, :]}")
    print(f"valid: {valid_np[0, 0, 0, 0, :]}")
    print(f"|a|: {np.abs(a_complex[0, 0, 0, 0, :])}")
    print(f"interactions (all depths) for that pair:\n{inter_np[:, 0, 0, 0, 0, :]}")
    print("=" * 70)

    return {
        "total_valid_paths": int(valid_np.sum()),
        "valid_per_pair": valid_per_pair,
        "interaction_counts": interaction_counts,
        "los_only_count": int(is_los_only.sum()),
    }

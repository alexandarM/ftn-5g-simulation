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
import mitsuba as mi
from sionna.rt import PathSolver, Camera
from sionna.rt.constants import InteractionType


def make_bird_eye_camera(position=(200.0, -155.0, 800.0), look_at=(0.0, 0.0, 0.0)):
    """Creates a bird's‑eye camera above the campus for scene/path previews."""
    return Camera(
        position=np.array(position, dtype=np.float32),
        look_at=np.array(look_at, dtype=np.float32),
    )


def render_scene(scene, camera, resolution=(1024, 1024), num_samples=1024, paths=None,
                  max_paths=8000, path_subsample_seed=41):
    """
    Renders a 3D view of the scene, optionally with `paths` overlaid.
    Corresponds to cells 66 and 70.

    WHY `max_paths` EXISTS ()
    -----------------------------------------------------
    Sionna RT's renderer does NOT draw all overlaid paths as one cheap
    geometry. Internally (`sionna.rt.renderer.get_overlay_scene` ->
    `paths_to_segments`), it turns EVERY valid (rx, tx, path) triple into
    its own sequence of individual Mitsuba `cylinder` shapes (one per
    bounce), each with its own `area`-light emitter, and compiles ALL of
    them into a brand-new Mitsuba scene via `mi.load_dict(...)` on every
    single render call. There is no built-in limit or subsampling.

    If the number of valid paths exceeds `max_paths`, this function
    temporarily thins `paths`' internal validity mask down to a random
    subset of `max_paths` entries (spread across all rx/tx pairs, not just
    the first N so the overlay still looks representative), renders with
    that thinned mask, and then ALWAYS restores the original mask before
    returning (even on error) - so `paths` is left completely unmodified
    for anything called afterwards (e.g. `diagnose_paths`,
    `plot_channel_metrics`, which need the full path set).

    Parameters
    ----------
    max_paths : int
        Maximum number of (rx, tx, path) triples to keep for the overlay
        render when `paths` is given. 5000-10000 is normally enough to see
        the ray pattern without exhausting memory; lower it further (e.g.
        1000-2000) if you still see high RAM after this. Ignored if
        `paths` is None or the number of valid paths is already
        <= max_paths.
    path_subsample_seed : int
        Seed for the random subsample, for reproducible figures.
    """
    kwargs = dict(camera=camera, resolution=list(resolution), num_samples=num_samples)

    if paths is None:
        return scene.render(**kwargs)

    valid_np = np.array(paths.valid).astype(bool)
    n_valid = int(valid_np.sum())

    if n_valid <= max_paths:
        kwargs["paths"] = paths
        return scene.render(**kwargs)

    flat = valid_np.reshape(-1)
    true_idx = np.flatnonzero(flat)
    rng = np.random.default_rng(path_subsample_seed)
    keep_idx = rng.choice(true_idx, size=max_paths, replace=False)
    thinned_flat = np.zeros_like(flat)
    thinned_flat[keep_idx] = True
    thinned = thinned_flat.reshape(valid_np.shape)

    # `Paths.valid` has no public setter, so we swap the private backing
    # field directly -- restored in `finally` no matter what happens.
    original_valid = paths._valid
    try:
        paths._valid = mi.TensorXb(thinned)
        print(f"[render_scene] {n_valid} valid paths exceed max_paths={max_paths} "
              f"-- overlaying a random subset of {max_paths} for THIS RENDER ONLY. "
              f"(Rendering all {n_valid} would compile {n_valid}+ individual Mitsuba "
              f"shapes into the overlay scene -- that compilation, not PathSolver, "
              f"is what causes the OOM crash. `paths` is restored to its full set "
              f"immediately after this render.)")
        kwargs["paths"] = paths
        fig = scene.render(**kwargs)
    finally:
        paths._valid = original_valid

    return fig


def run_path_solver(scene, max_depth=3, los=True, specular_reflection=True,
                     diffuse_reflection=False, refraction=True,
                     synthetic_array=False, seed=41, check_diffusion_materials=True,
                     samples_per_src=1_000_000, max_num_paths_per_src=1_000_000):
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
    samples_per_src : int
        Number of candidate rays traced PER TRANSMITTER during candidate
        generation (Sionna RT default: 1 000 000). This is the main memory/
        compute lever for diffuse scattering: enabling `diffuse_reflection`
        on large, highly‑visible surfaces (e.g. the ground "plane" of a
        campus/street scene) causes a combinatorial explosion in the number
        of diffuse candidate interactions per source. Sionna's internal
        candidate‑path buffer is indexed with a 32‑bit type, so at the
        default 1e6 this can overflow ("... exceeds max value of 2^32 ...")
        as soon as diffuse reflection is enabled on a large-area object.
        Lowering THIS parameter (not `max_depth`) is the correct way to fix
        that: it reduces the candidate-ray density without reducing the
        number of bounces a found path can have, so multi-bounce realism
        (max_depth=3 or higher) can be preserved for a closer-to-1:1 digital
        twin. Typical starting point when diffuse reflection is enabled on
        large surfaces: 1e5-5e5.
    max_num_paths_per_src : int
        Maximum size of the candidate‑path buffer per transmitter (Sionna RT
        default: 1_000_000). Works together with `samples_per_src` to bound
        the same 2^32 buffer‑size limit; lower it in tandem if you still hit
        the overflow after reducing `samples_per_src` alone.
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
        samples_per_src=samples_per_src,
        max_num_paths_per_src=max_num_paths_per_src,
    )
    return paths


def diagnose_paths(paths):
    """
    Detailed diagnostic printout of the `paths` object: array shapes (a, tau,
    valid, interactions), total valid paths, statistics per (rx, tx) pair,
    interaction type distribution (NONE/SPECULAR/DIFFUSE/REFRACTION/DIFFRACTION),
    number of pure LoS paths, and example data for the first (rx, tx) pair.

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

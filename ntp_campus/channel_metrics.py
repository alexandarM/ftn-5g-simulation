"""
channel_metrics.py
====================

Analysis of PathSolver results at the channel level: impulse response (CIR),
frequency response (CFR), discrete taps, RSS per user, number of paths per user,
path loss vs distance, RMS delay spread, and distribution of interaction types.

The main function `plot_channel_metrics` generates 8 plots and saves them in the
`figures/` folder with a `model_name` prefix – useful for comparing different
configurations (1x1 vs 12x1 arrays, with/without diffusion, etc.).
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sionna.rt import subcarrier_frequencies

from . import config


def setup_plot_style():
    """
    Applies a consistent visual style for all plots (seaborn if available,
    otherwise a matplotlib fallback) and ensures the `figures/` directory exists.
    """
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_palette("viridis")
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["axes.prop_cycle"] = plt.cycler(
            color=["#2E86C1", "#28B463", "#E67E22", "#8E44AD", "#E74C3C", "#1ABC9C"]
        )


def _rms_delay_spread(tau_ns, a_abs):
    """
    Computes RMS delay spread (a measure of channel time dispersion) from
    delays `tau_ns` [ns] and corresponding amplitudes `a_abs`, weighted by |a|^2.

    RMS delay spread = sqrt(E[tau^2] - E[tau]^2), where expectations are
    power-weighted over all paths.
    """
    power = a_abs ** 2
    if power.sum() == 0:
        return np.nan
    tau_mean = np.sum(tau_ns * power) / np.sum(power)
    tau2_mean = np.sum(tau_ns ** 2 * power) / np.sum(power)
    return np.sqrt(max(tau2_mean - tau_mean ** 2, 0))


def _compute_rss_and_paths_per_user(a_complex, valid_np, scene):
    """
    Computes RSS [dBm] per (rx, tx) pair (coherent across TX ports,
    incoherent across RX ports and paths, normalised by the number of TX ports),
    then for each user selects the best (serving) transmitter and returns
    RSS and path count for that pair.

    Returns
    -------
    tuple (rss_dbm (num_rx, num_tx), best_tx_idx, best_rss, valid_rss, best_paths)
    """
    tx_names = list(scene.transmitters.keys())
    num_rx = a_complex.shape[0]
    num_tx = len(tx_names)
    num_tx_ant = a_complex.shape[3]

    rss_dbm = np.full((num_rx, num_tx), -np.inf)

    for rx_idx in range(num_rx):
        for tx_idx in range(num_tx):
            a_rx_tx = a_complex[rx_idx, :, tx_idx, :, :]      # (rx_ant, tx_ant, paths)
            valid_rx_tx = valid_np[rx_idx, :, tx_idx, :, :]

            total_power = 0.0
            for p in range(a_rx_tx.shape[2]):
                if valid_rx_tx[:, :, p].any():
                    a_path = a_rx_tx[:, :, p]
                    a_coherent = np.sum(a_path, axis=1)  # coherent across TX axis
                    total_power += np.sum(np.abs(a_coherent) ** 2)  # incoherent across RX axis

            total_power /= num_tx_ant

            if total_power > 0:
                p_tx_dbm = float(np.asarray(scene.transmitters[tx_names[tx_idx]].power_dbm).flatten()[0])
                rss_dbm[rx_idx, tx_idx] = 10 * np.log10(total_power) + p_tx_dbm

    best_tx_idx = np.argmax(rss_dbm, axis=1)
    best_rss = np.max(rss_dbm, axis=1)
    valid_rss = best_rss[best_rss > -np.inf]

    paths_per_pair = valid_np.sum(axis=(1, 3, 4))
    best_paths = paths_per_pair[np.arange(num_rx), best_tx_idx]

    return rss_dbm, best_tx_idx, best_rss, valid_rss, best_paths


def _cfr_for_pairs(a_cir, tau_cir, frequencies, pair_idx):
    """
    Computes h(f) = sum_i a_i * exp(-j*2*pi*f*tau_i) for a small list of
    (rx, rx_ant, tx, tx_ant) index tuples, using CIR output already
    computed by `paths.cir(normalize_delays=True, out_type="numpy")`.

    Parameters
    ----------
    a_cir : np.ndarray [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        Complex baseband-equivalent CIR, from `paths.cir(...)`.
    tau_cir : np.ndarray [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        Path delays [s]; invalid/padding slots are -1 (Sionna's sentinel
        when `normalize_delays=True`).
    frequencies : np.ndarray [num_freq]
    pair_idx : sequence of (ir, ira, it, ita) tuples

    Returns
    -------
    np.ndarray [len(pair_idx), num_freq]
    """
    n_pairs = len(pair_idx)
    h = np.zeros((n_pairs, len(frequencies)), dtype=np.complex64)
    for k, (ir, ira, it, ita) in enumerate(pair_idx):
        a_p = a_cir[ir, ira, it, ita, :, 0]
        tau_p = tau_cir[ir, ira, it, ita, :]
        mask = tau_p > -0.5  # exclude invalid/padding slots (sentinel tau=-1)
        a_p, tau_p = a_p[mask], tau_p[mask]
        if a_p.size == 0:
            continue
        phase = -2.0 * np.pi * np.outer(tau_p, frequencies)  # [n_valid_paths, num_freq]
        h[k] = np.sum(a_p[:, None] * np.exp(1j * phase), axis=0)

    h_abs = np.abs(h)
    energy = np.mean(h_abs ** 2, axis=1, keepdims=True)
    energy[energy == 0] = 1.0
    return h_abs / np.sqrt(energy)


def _taps_for_pairs(a_cir, tau_cir, tap_indices, bandwidth, pair_idx):
    """
    Computes h[l] = sum_i a_i * sinc(l - bandwidth*tau_i) for a small list
    of (rx, rx_ant, tx, tx_ant) index tuples - the manual, per-pair
    equivalent of `sionna.rt.Paths.taps` (see `_cfr_for_pairs` docstring
    above for why this is needed;
    """
    n_pairs = len(pair_idx)
    h = np.zeros((n_pairs, len(tap_indices)), dtype=np.complex64)
    for k, (ir, ira, it, ita) in enumerate(pair_idx):
        a_p = a_cir[ir, ira, it, ita, :, 0]
        tau_p = tau_cir[ir, ira, it, ita, :]
        mask = tau_p > -0.5
        a_p, tau_p = a_p[mask], tau_p[mask]
        if a_p.size == 0:
            continue
        g = np.sinc(tap_indices[None, :] - bandwidth * tau_p[:, None])  # [n_valid_paths, n_taps]
        h[k] = np.sum(a_p[:, None] * g, axis=0)

    h_abs = np.abs(h)
    energy = np.sum(h_abs ** 2, axis=1, keepdims=True)  # Sionna: total block energy = 1, not mean
    energy[energy == 0] = 1.0
    return h_abs / np.sqrt(energy)


def plot_channel_metrics(paths, scene, model_name="current", save=True, tx_gain_dbi=17.0):
    """
    Generates a complete set of 8 plots for channel analysis, based on
    `PathSolver` results (`paths`) and the scene.

    Plots (in order of generation):
        1. CIR grid (3x3)          – impulse responses for 9 random (rx,tx) pairs
        2. CFR (3 pairs)           – frequency response |H(f)|
        3. TAPS (3 pairs)          – discrete channel taps
        4. RSS histogram           – distribution of received power per user
        5. Path count histogram    – how many multipath components each user sees
        6. Path loss vs distance   – scatter plot of RSS versus TX-RX distance
        7. RMS delay spread hist.  – channel time dispersion
        8. Interaction distribution – counts of LOS/SPECULAR/DIFFUSE/REFRACTION/DIFFRACTION

    Parameters
    ----------
    paths : sionna.rt.Paths
        Output from `path_solver.run_path_solver`.
    scene : sionna.rt.Scene
    model_name : str
        Configuration name, used as a suffix in saved filenames.
    save : bool
        Whether to save plots to the `figures/` folder.
    tx_gain_dbi : float
        Transmitter antenna gain [dBi] (informational; currently not used directly
        because it's already embedded in `paths.a`, but kept for transparency).

    Returns
    -------
    dict
        {"rss": ndarray, "paths_per_user": ndarray, "rms_delay_spread": ndarray or None}
    """
    setup_plot_style()
    colors = config.COLORS

    print("\n" + "=" * 70)
    print(f"GENERATING PLOTS FOR: {model_name}")
    print("=" * 70)

    # --- 1. Extract basic data from the paths object ---
    a_real, a_imag = paths.a
    a_complex = np.array(a_real) + 1j * np.array(a_imag)
    valid_np = np.array(paths.valid).astype(bool)
    tau_np = np.array(paths.tau)

    # CIR (normalised by delay of first path)
    a_cir, tau_cir = paths.cir(normalize_delays=True, out_type="numpy")
    a_cir_abs = np.abs(a_cir)[..., 0]  # remove time_step axis (static channel)

    # --- RSS and path counts per user (best serving TX) ---
    rss_dbm, best_tx_idx, best_rss, valid_rss, best_paths = \
        _compute_rss_and_paths_per_user(a_complex, valid_np, scene)

    num_rx = a_complex.shape[0]
    num_tx = a_complex.shape[2]
    tx_names = list(scene.transmitters.keys())

    # --- 2. Find valid (rx,tx) pairs for CIR/CFR/TAPS display ---
    # (this now happens BEFORE the CFR/TAPS computation below, since CFR/TAPS
    # are only computed for these specific selected pairs -- see comments there)
    active_per_pair = (a_cir_abs > 0).sum(axis=-1)
    candidates = np.argwhere(active_per_pair >= 3)
    if len(candidates) == 0:
        print("Warning: No pairs with enough paths for CIR plot.")
        candidates = np.argwhere(active_per_pair >= 1)

    n_plots = min(9, len(candidates))
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    selected_idx = candidates[rng.choice(len(candidates), size=n_plots, replace=False)]

    # CFR (frequency domain, derived from CIR via DFT). Computed manually,
    # only for the (up to 3) pairs that PLOT 2 below actually plots -- see
    # `_cfr_for_pairs` docstring for exactly why (avoids a Dr.Jit dense-
    # tensor overflow that a full `paths.cfr()` call hits on large
    # TX arrays with diffuse reflection).
    num_subcarriers = 1024
    frequencies = subcarrier_frequencies(num_subcarriers, 30e3)
    h_abs_pairs = _cfr_for_pairs(a_cir, tau_cir, frequencies, selected_idx[:3])

    # TAPS (discretised channel for a given bandwidth and tap index range).
    # Same reasoning and same fix as CFR above -- see `_taps_for_pairs`.
    l_min, l_max, bandwidth = -6, 100, 100e6
    tap_indices = np.arange(l_min, l_max + 1)
    taps_abs_pairs = _taps_for_pairs(a_cir, tau_cir, tap_indices, bandwidth, selected_idx[:3])

    # --- 3. PLOT 1: CIR GRID (3x3) ---
    print("\n[1] Generating CIR grid...")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes_flat = axes.flat

    for ax, (ir, ira, it, ita) in zip(axes_flat, selected_idx):
        t = tau_np[ir, ira, it, ita, :] / 1e-9  # seconds -> nanoseconds
        a_abs = a_cir_abs[ir, ira, it, ita, :]
        mask = t > -1e8  # filter out "empty" invalid slots

        ax.stem(t[mask], a_abs[mask], linefmt="b-", markerfmt="bo", basefmt=" ")
        n_paths = mask.sum()
        ax.set_title(f"rx={ir}, tx={it} ({n_paths} paths)", fontsize=9)
        ax.set_xlabel(r"$\tau$ [ns]", fontsize=8)
        ax.set_ylabel(r"$|a|$", fontsize=8)
        ax.grid(alpha=0.3)

    for i in range(n_plots, 9):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig(f"figures/cir_grid_{model_name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: figures/cir_grid_{model_name}.png")
    plt.show()

    # --- 4. PLOT 2: CFR for 3 pairs ---
    print("\n[2] Generating CFR plots...")
    n_cfr = min(3, len(selected_idx))
    fig, axes = plt.subplots(1, n_cfr, figsize=(5 * n_cfr, 4))
    if n_cfr == 1:
        axes = [axes]

    for ax, (ir, ira, it, ita), h_example in zip(axes, selected_idx[:n_cfr], h_abs_pairs):
        ax.plot(h_example, color=colors["teal"], linewidth=1.2)
        ax.set_xlabel("Subcarrier index")
        ax.set_ylabel(r"$|h_{freq}|$")
        ax.set_title(f"rx={ir}, tx={it}")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(f"figures/cfr_{model_name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: figures/cfr_{model_name}.png")
    plt.show()

    # --- 5. PLOT 3: TAPS for 3 pairs ---
    print("\n[3] Generating TAPS plots...")
    n_taps = min(3, len(selected_idx))
    fig, axes = plt.subplots(1, n_taps, figsize=(5 * n_taps, 4))
    if n_taps == 1:
        axes = [axes]

    for ax, (ir, ira, it, ita), taps_example in zip(axes, selected_idx[:n_taps], taps_abs_pairs):
        ax.stem(tap_indices, taps_example, linefmt="r-", markerfmt="ro", basefmt=" ")
        ax.set_xlabel(r"Tap index $\ell$")
        ax.set_ylabel(r"$|h[\ell]|$")
        ax.set_title(f"rx={ir}, tx={it}")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(f"figures/taps_{model_name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: figures/taps_{model_name}.png")
    plt.show()

    # --- 6. PLOT 4: RSS HISTOGRAM ---
    print("\n[4] Generating RSS histogram...")
    plt.figure(figsize=(10, 6))
    plt.hist(valid_rss, bins=30, alpha=0.7, color=colors["blue"], edgecolor=colors["dark"])
    plt.axvline(np.mean(valid_rss), color=colors["red"], linestyle="--", linewidth=2,
                label=f"Mean = {np.mean(valid_rss):.1f} dBm")
    plt.xlabel("RSS [dBm]", fontsize=12)
    plt.ylabel("Number of users", fontsize=12)
    plt.title(f"RSS distribution - {model_name}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    if save:
        plt.savefig(f"figures/rss_histogram_{model_name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: figures/rss_histogram_{model_name}.png")
    plt.show()

    # --- 7. PLOT 5: HISTOGRAM OF PATH COUNTS ---
    print("\n[5] Generating path count histogram...")
    plt.figure(figsize=(10, 6))
    n_bins = 40
    bin_edges = np.linspace(0, best_paths.max() + 1, n_bins + 1)
    plt.hist(best_paths, bins=bin_edges, alpha=0.7, color=colors["green"], edgecolor=colors["dark"])
    plt.axvline(np.mean(best_paths), color=colors["red"], linestyle="--", linewidth=2,
                label=f"Mean = {np.mean(best_paths):.1f}")
    plt.xlabel("Number of paths", fontsize=12)
    plt.ylabel("Number of users", fontsize=12)
    plt.title(f"Paths per user - {model_name}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    if save:
        plt.savefig(f"figures/paths_histogram_{model_name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: figures/paths_histogram_{model_name}.png")
    plt.show()

    # --- 8. PLOT 6: PATH LOSS VS DISTANCE ---
    print("\n[6] Generating Path Loss plot...")
    rx_positions = []
    for rx_name in sorted(scene.receivers.keys()):
        rx_positions.append(np.array(scene.receivers[rx_name].position))
    rx_positions = np.array(rx_positions)

    distances, rss_db = [], []
    for rx_idx in range(num_rx):
        for tx_idx in range(num_tx):
            if rss_dbm[rx_idx, tx_idx] > -np.inf:
                dist = np.linalg.norm(rx_positions[rx_idx] -
                                       np.array(scene.transmitters[tx_names[tx_idx]].position))
                distances.append(dist)
                rss_db.append(rss_dbm[rx_idx, tx_idx])

    plt.figure(figsize=(10, 6))
    plt.scatter(distances, rss_db, s=30, alpha=0.6, color=colors["purple"],
                edgecolor=colors["dark"], linewidth=0.5)
    plt.xlabel("TX-RX distance [m]", fontsize=12)
    plt.ylabel("RSS [dBm]", fontsize=12)
    plt.title(f"Path Loss - {model_name}", fontsize=14)
    plt.grid(alpha=0.3)
    if save:
        plt.savefig(f"figures/path_loss_{model_name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: figures/path_loss_{model_name}.png")
    plt.show()

    # --- 9. PLOT 7: RMS DELAY SPREAD ---
    print("\n[7] Generating RMS Delay Spread histogram...")
    ds_values = []
    for ir in range(num_rx):
        for it in range(num_tx):
            v = valid_np[ir, 0, it, 0, :]
            if v.sum() < 2:
                continue
            t_ns = tau_np[ir, 0, it, 0, v] * 1e9
            a_v = a_cir_abs[ir, 0, it, 0, v]
            ds_values.append(_rms_delay_spread(t_ns, a_v))

    ds_values = np.array(ds_values)
    ds_values = ds_values[~np.isnan(ds_values)]

    if len(ds_values) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(ds_values, bins=30, alpha=0.7, color=colors["orange"], edgecolor=colors["dark"])
        plt.axvline(np.mean(ds_values), color=colors["red"], linestyle="--", linewidth=2,
                    label=f"Mean = {np.mean(ds_values):.1f} ns")
        plt.xlabel("RMS delay spread [ns]", fontsize=12)
        plt.ylabel("Number of (rx,tx) pairs", fontsize=12)
        plt.title(f"RMS Delay Spread - {model_name}", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        if save:
            plt.savefig(f"figures/rms_delay_spread_{model_name}.png", dpi=150, bbox_inches="tight", facecolor="white")
            print(f"  Saved: figures/rms_delay_spread_{model_name}.png")
        plt.show()

    # --- 10. PLOT 8: DISTRIBUTION OF INTERACTION TYPES ---
    print("\n[8] Generating interaction distribution...")
    inter_np = np.array(paths.interactions).astype(np.uint32)
    unique_vals, counts = np.unique(inter_np, return_counts=True)
    type_names = {0: "NONE", 1: "SPECULAR", 2: "DIFFUSE", 4: "REFRACTION", 8: "DIFFRACTION"}

    labels, values = [], []
    for v, c in zip(unique_vals, counts):
        labels.append(type_names.get(int(v), f"unknown({v})"))
        values.append(c)

    interaction_colors = [colors["gray"], colors["blue"], colors["green"], colors["orange"], colors["red"]]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, alpha=0.8, color=interaction_colors[:len(labels)], edgecolor=colors["dark"])
    plt.ylabel("Number of interactions", fontsize=12)
    plt.title(f"Interaction type distribution - {model_name}", fontsize=14)
    plt.grid(alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                  f"{val}", ha="center", va="bottom", fontsize=9, rotation=0)

    if save:
        plt.savefig(f"figures/interactions_{model_name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: figures/interactions_{model_name}.png")
    plt.show()

    # --- 11. Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total valid paths: {valid_np.sum()}")
    print(f"  Average paths per user: {np.mean(best_paths)}")
    print(f"  RSS (mean): {np.mean(valid_rss)} dBm")
    print(f"  RSS (std): {np.std(valid_rss)} dBm")
    if len(ds_values) > 0:
        print(f"  RMS delay spread (mean): {np.mean(ds_values)} ns")
    print(f"  Saved 8 plots in figures/ folder")
    print("=" * 70)

    return {
        "rss": valid_rss,
        "paths_per_user": best_paths,
        "rms_delay_spread": ds_values if len(ds_values) > 0 else None,
    }

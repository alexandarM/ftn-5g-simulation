"""
verification.py
=================

Orchestrates verification of the "air3278_broadcast" antenna pattern on an empty
(no buildings) scene. This way the measured values (V‑BW, H‑BW, F/B, SLL) can
be compared directly against datasheet targets, without occlusion or reflections
from campus geometry.

This corresponds to cells 15, 20‑24 from the original notebook, which were
scripted at global scope; here they're wrapped into functions that return
results and optionally plot figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from sionna.rt import Transmitter, PlanarArray

from . import config
from .scene_setup import load_empty_scene
from .antenna_measurements import (
    measure_vbw_simple,
    measure_hbw_envelope,
    measure_front_to_back_ratio_batch,
    measure_side_lobe_level_vertical,
)


def build_test_transmitter(clean_scene, tx_name="TX_CLEAN_TEST",
                            position=(0.0, 0.0, 31.8),
                            tx_yaw_deg=20.0, tx_tilt_deg=4.0,
                            tx_power_dbm=None):
    """
    Sets `tx_array` (with the registered `air3278_broadcast` pattern) and adds
    one test transmitter to the empty scene. Any yaw/tilt angles are valid for
    pattern verification since the scene has no geometry that would constrain
    the choice.

    Note: orientation here is NOT computed via `compass_azimuth_to_yaw`
    (compass convention), but directly as `90° - tx_yaw_deg`, because
    `tx_yaw_deg` here is already a "mathematical" azimuth used only for testing.

    Returns
    -------
    sionna.rt.Transmitter
        The added test transmitter (also present in `clean_scene.transmitters`).
    """
    if tx_power_dbm is None:
        tx_power_dbm = config.TX_POWER_DBM_DEFAULT

    clean_scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="air3278_broadcast", polarization="cross",
    )

    if tx_name in clean_scene.transmitters:
        clean_scene.remove(tx_name)

    tx = Transmitter(
        name=tx_name,
        position=list(position),
        orientation=[
            float(np.deg2rad(90.0 - tx_yaw_deg)),
            float(np.deg2rad(tx_tilt_deg)),
            0.0,
        ],
        power_dbm=tx_power_dbm,
    )
    clean_scene.add(tx)
    print(f"Added test transmitter '{tx_name}' on empty scene, "
          f"yaw={tx_yaw_deg}°, tilt={tx_tilt_deg}°")
    return tx


def run_vbw_hbw_verification(clean_scene, tx_name="TX_CLEAN_TEST", plot=True):
    """
    Runs V‑BW and H‑BW measurements on the empty scene and optionally plots
    diagrams with the ‑3 dB threshold. Corresponds to cell 15.

    Returns
    -------
    dict with keys: vbw, v_peak_el, v_elevs, v_eirp, hbw, h_peak_phi,
    h_bin_centers, h_bin_max_eirp.
    """
    print("\n" + "=" * 60)
    print("V-BW TEST (empty scene)")
    print("=" * 60)
    vbw, v_peak_el, v_elevs, v_eirp = measure_vbw_simple(clean_scene, tx_name)
    print(f"V-BW = {vbw}°  (peak at world elevation {v_peak_el}°, "
          f"target: {config.V_BW_DATASHEET}°±1°)")

    print("\n" + "=" * 60)
    print("H-BW TEST (empty scene)")
    print("=" * 60)
    hbw, h_peak_phi, bc, bmax = measure_hbw_envelope(clean_scene, tx_name)
    print(f"H-BW = {hbw}°  (peak at local phi_t={h_peak_phi}°, "
          f"target: {config.H_BW_DATASHEET}°±5°)")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(v_elevs, v_eirp, "o-", markersize=3)
        axes[0].axhline(np.nanmax(v_eirp) - 3, color="orange", linestyle=":", label="-3dB")
        axes[0].set_xlabel("Elevation [°]")
        axes[0].set_ylabel("EIRP [dBm]")
        axes[0].set_title(f"V-cut (empty scene), V-BW={vbw}°")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(bc, bmax, "o-", markersize=3)
        axes[1].axhline(bmax.max() - 3, color="orange", linestyle=":", label="-3dB")
        axes[1].set_xlabel("Local phi_t [°]")
        axes[1].set_ylabel("Max EIRP [dBm]")
        axes[1].set_title(f"H-cut envelope (empty scene), H-BW={hbw}°")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {"vbw": vbw, "v_peak_el": v_peak_el, "v_elevs": v_elevs, "v_eirp": v_eirp,
            "hbw": hbw, "h_peak_phi": h_peak_phi, "h_bin_centers": bc, "h_bin_max_eirp": bmax}


def run_fbr_verification(clean_scene, tx_name="FBR_TEST_TX", radius=100.0,
                          save_fig_path="figures/fbr_test.png", plot=True):
    """
    Runs the F/B measurement (batch variant) on the empty scene and optionally
    plots a diagram comparing against the datasheet target.

    Returns the result of `measure_front_to_back_ratio_batch`.
    """
    result = measure_front_to_back_ratio_batch(clean_scene, tx_name, radius=radius)

    if plot:
        eirp_back_arr = np.nanmax(result["eirp_back_grid"], axis=0)

        plt.figure(figsize=(8, 5))
        plt.plot(result["back_offsets"] - 180.0, eirp_back_arr, "o-",
                  color="#E74C3C", label="Rear hemisphere (max over elevation)")
        plt.axhline(result["eirp_front"], color="#2E86C1", linestyle="--",
                     label=f'Front (boresight) = {result["eirp_front"]} dBm')
        plt.axhline(result["eirp_front"] - config.FBR_DATASHEET, color="gray", linestyle=":",
                     label=f"Target: front − {config.FBR_DATASHEET} dB")
        plt.xlabel("Offset from 180° [°]")
        plt.ylabel("EIRP [dBm]")
        plt.title(f'F/B test — measured {result["fbr_measured"]} dB (target {config.FBR_DATASHEET} dB)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_fig_path:
            plt.savefig(save_fig_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.show()

    return result


def run_sll_verification(clean_scene, tx_name="FBR_TEST_TX", radius=100.0,
                          span_deg=60.0, n_points=400):
    """
    Runs the vertical SLL measurement on the empty scene (reusing the same
    test transmitter as the F/B test). Corresponds to cells 22‑24.

    Returns the result of `measure_side_lobe_level_vertical`.
    """
    return measure_side_lobe_level_vertical(clean_scene, tx_name, radius=radius,
                                             span_deg=span_deg, n_points=n_points)


def run_full_antenna_verification(tx_yaw_deg=20.0, tx_tilt_deg=4.0, tx_power_dbm=None):
    """
    Combines all verification steps (V‑BW, H‑BW, F/B, SLL) into a single call:
    creates an empty scene, sets up a test transmitter, runs all measurements,
    and prints/plots the results. Convenient for calling from `main.py` as a
    self‑contained "check that the antenna pattern matches the datasheet" step
    before using the same pattern on the full campus scene.

    Returns
    -------
    dict with keys 'vbw_hbw', 'fbr', 'sll'.
    """
    clean_scene = load_empty_scene()
    build_test_transmitter(clean_scene, tx_name="TX_CLEAN_TEST",
                            tx_yaw_deg=tx_yaw_deg, tx_tilt_deg=tx_tilt_deg,
                            tx_power_dbm=tx_power_dbm)
    vbw_hbw = run_vbw_hbw_verification(clean_scene, tx_name="TX_CLEAN_TEST")

    # Important: F/B and SLL tests must run on a FRESH empty scene with only
    # one transmitter ("FBR_TEST_TX"), not on `clean_scene` above (which still
    # contains "TX_CLEAN_TEST"). Functions like `measure_vbw_simple` and
    # `measure_side_lobe_level_vertical` do not explicitly select a transmitter
    # index (unlike the F/B functions that use `measure_eirp_at_direction`).
    # They sum power across ALL transmitters in the scene. If two transmitters
    # remained, the pattern would be smeared and SLL detection would miss the
    # side lobe (or F/B measurements would be wrong).
    fbr_scene = load_empty_scene()
    build_test_transmitter(fbr_scene, tx_name="FBR_TEST_TX",
                            tx_yaw_deg=tx_yaw_deg, tx_tilt_deg=tx_tilt_deg,
                            tx_power_dbm=tx_power_dbm)
    fbr = run_fbr_verification(fbr_scene, tx_name="FBR_TEST_TX")
    sll = run_sll_verification(fbr_scene, tx_name="FBR_TEST_TX")

    return {"vbw_hbw": vbw_hbw, "fbr": fbr, "sll": sll}

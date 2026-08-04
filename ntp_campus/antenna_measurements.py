"""
antenna_measurements.py
=========================

Routines for "electromagnetic measurement" of the simulated antenna pattern.
Places probe receivers on a circle or grid around the transmitter and uses
PathSolver with `max_depth=0` (line-of-sight only) to compute EIRP per direction.

From the resulting radiation diagram we extract:
    - V-BW  (measure_vbw_simple)          – vertical -3 dB beamwidth
    - H-BW  (measure_hbw_envelope)        – horizontal -3 dB beamwidth
    - F/B   (measure_front_to_back_ratio_*) – front-to-back ratio
    - SLL   (measure_side_lobe_level_vertical) – vertical side-lobe level

These functions are generic (don't depend directly on config/antenna_pattern),
but in practice they're used to verify the `air3278_broadcast` pattern defined
elsewhere.

EIRP formula note
------------------
Complex path gains are first summed coherently across TX antenna ports (array
factor), then power is summed incoherently across RX antenna ports and paths.
The result is normalised by the number of TX ports, converted to dB, and
combined with FSPL and per-port TX power to get EIRP for that direction.
"""

import numpy as np
from sionna.rt import Receiver, PlanarArray, PathSolver

from .antenna_pattern import to_scalar
from . import config


# =========================================================================
# 1. VERTICAL BEAMWIDTH (V-BW)
# =========================================================================
def measure_vbw_simple(sc, tx_name, radius=100.0, tx_power_dbm=53.0,
                        span_deg=60.0, n_points=150):
    """
    Measures vertical -3 dB beamwidth. Scans EIRP along the vertical plane
    through the boresight (fixed azimuth = transmitter yaw), elevation from
    -span_deg to +span_deg. Returns beamwidth, peak elevation, elevation array
    and corresponding EIRP.
    """
    tx = sc.transmitters[tx_name]
    tx.power_dbm = tx_power_dbm
    p_tx_dbm = to_scalar(tx.power_dbm)
    tx_pos = np.array(tx.position, dtype=np.float64).reshape(-1)[:3]
    tx_ori = np.array(tx.orientation, dtype=np.float64).reshape(-1)
    tx_yaw = to_scalar(tx_ori[0])

    # Clear any leftover probes
    for rx_name in list(sc.receivers.keys()):
        sc.remove(rx_name)

    # Place probes along an arc
    elevations = np.linspace(-span_deg, span_deg, n_points)
    positions = []
    for el in elevations:
        el_rad = np.radians(-el)
        dx = radius * np.cos(el_rad) * np.cos(tx_yaw)
        dy = radius * np.cos(el_rad) * np.sin(tx_yaw)
        dz = radius * np.sin(el_rad)
        pos = [float(tx_pos[0] + dx), float(tx_pos[1] + dy), float(tx_pos[2] + dz)]
        positions.append(pos)
        sc.add(Receiver(name=f"vbw_probe_{len(positions)}", position=pos))

    # Isotropic RX array, 1x1 – no extra gain on receive
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                               horizontal_spacing=0.5, pattern="iso", polarization="cross")
    solver = PathSolver()
    paths = solver(scene=sc, max_depth=0)  # LOS only – clean pattern
    a_np = np.array(paths.a)

    n = len(positions)
    rx_axis = next((i for i, s in enumerate(a_np.shape) if s == n), 0)
    lambda_m = config.SPEED_OF_LIGHT_M_S / sc.frequency

    # Original shape: (num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, ...)
    tx_ant_axis_original = 4
    num_tx_ant = a_np.shape[tx_ant_axis_original]

    eirp = []
    for k in range(n):
        slices = [slice(None)] * len(a_np.shape)
        slices[rx_axis] = k
        a_rx = a_np[tuple(slices)]

        # After removing rx_axis, TX antenna axis index might shift
        tx_ant_axis = tx_ant_axis_original if rx_axis > tx_ant_axis_original \
            else tx_ant_axis_original - 1

        # Coherent sum over TX ports
        a_coherent = np.sum(a_rx, axis=tx_ant_axis)
        # Incoherent sum over remaining dims, normalise
        p_lin = to_scalar(np.sum(np.abs(a_coherent) ** 2)) / float(num_tx_ant)

        if p_lin > 0:
            dist = to_scalar(np.linalg.norm(np.array(positions[k]) - tx_pos))
            fspl_db = 20.0 * np.log10(4 * np.pi * dist / lambda_m)
            eirp.append(to_scalar(10.0 * np.log10(p_lin) + fspl_db + p_tx_dbm))
        else:
            eirp.append(float(np.nan))

    # Clean up probes
    for rx_name in list(sc.receivers.keys()):
        sc.remove(rx_name)

    eirp = np.array(eirp, dtype=np.float64)
    valid = ~np.isnan(eirp)
    print(f"  [debug V] valid points: {valid.sum()}/{n}")
    tv, yv = elevations[valid], eirp[valid]
    peak_idx = int(np.argmax(yv))
    thresh = yv[peak_idx] - 3.0

    def interp_crossing(i0, direction):
        """Linear interpolation of the point where EIRP crosses the threshold."""
        i = i0
        while True:
            j = i + direction
            if j < 0 or j >= len(yv):
                return tv[i]
            if yv[j] < thresh:
                frac = (yv[i] - thresh) / (yv[i] - yv[j])
                return tv[i] + frac * (tv[j] - tv[i])
            i = j

    left = interp_crossing(peak_idx, -1)
    right = interp_crossing(peak_idx, +1)
    return right - left, tv[peak_idx], tv, yv


# =========================================================================
# 2. 2D ANGLE SCAN (BASE FOR H-BW)
# =========================================================================
def scan_local_angles_and_eirp(sc, tx_name, world_el_offsets, world_az_offsets,
                                center_el_deg, center_az_deg,
                                radius=100.0, tx_power_dbm=53.0):
    """
    Scans EIRP over a 2D grid (elevation x azimuth) around a given centre.
    Returns local angles (theta_t, phi_t) as computed by Sionna – these
    account for the actual transmitter orientation, unlike world angles.
    """
    tx = sc.transmitters[tx_name]
    tx.power_dbm = tx_power_dbm
    p_tx_dbm = to_scalar(tx.power_dbm)
    tx_pos = np.array(tx.position, dtype=np.float64).reshape(-1)[:3]

    for rx_name in list(sc.receivers.keys()):
        sc.remove(rx_name)

    # Build 2D grid of probe positions
    EL, AZ = np.meshgrid(center_el_deg + world_el_offsets,
                          center_az_deg + world_az_offsets, indexing="ij")
    n_el, n_az = EL.shape
    positions = []
    for i in range(n_el):
        for j in range(n_az):
            el_rad = np.radians(-EL[i, j])
            az_rad = np.radians(AZ[i, j])
            dx = radius * np.cos(el_rad) * np.cos(az_rad)
            dy = radius * np.cos(el_rad) * np.sin(az_rad)
            dz = radius * np.sin(el_rad)
            pos = [float(tx_pos[0] + dx), float(tx_pos[1] + dy), float(tx_pos[2] + dz)]
            positions.append(pos)
            sc.add(Receiver(name=f"scan_probe_{len(positions)}", position=pos))

    n = len(positions)
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                               horizontal_spacing=0.5, pattern="iso", polarization="cross")
    solver = PathSolver()
    paths = solver(scene=sc, max_depth=0)

    a_np = np.array(paths.a)
    theta_t_np = np.degrees(np.array(paths.theta_t))
    phi_t_np = np.degrees(np.array(paths.phi_t))

    rx_axis_a = next((i for i, s in enumerate(a_np.shape) if s == n), 0)
    rx_axis_t = next((i for i, s in enumerate(theta_t_np.shape) if s == n), 0)

    tx_ant_axis_original = 4
    lambda_m = config.SPEED_OF_LIGHT_M_S / sc.frequency
    num_tx_ant = a_np.shape[tx_ant_axis_original]

    theta_list, phi_list, eirp_list = [], [], []
    for k in range(n):
        slices_a = [slice(None)] * len(a_np.shape)
        slices_a[rx_axis_a] = k
        a_rx = a_np[tuple(slices_a)]

        tx_ant_axis = tx_ant_axis_original if rx_axis_a > tx_ant_axis_original \
            else tx_ant_axis_original - 1

        a_coherent = np.sum(a_rx, axis=tx_ant_axis)
        p_lin = to_scalar(np.sum(np.abs(a_coherent) ** 2)) / float(num_tx_ant)

        slices_t = [slice(None)] * len(theta_t_np.shape)
        slices_t[rx_axis_t] = k
        th_vals = theta_t_np[tuple(slices_t)].reshape(-1)
        ph_vals = phi_t_np[tuple(slices_t)].reshape(-1)
        th_vals = th_vals[~np.isnan(th_vals)]
        ph_vals = ph_vals[~np.isnan(ph_vals)]

        if p_lin > 0 and len(th_vals) > 0:
            dist = to_scalar(np.linalg.norm(np.array(positions[k]) - tx_pos))
            fspl_db = 20.0 * np.log10(4 * np.pi * dist / lambda_m)
            eirp = to_scalar(10.0 * np.log10(p_lin) + fspl_db + p_tx_dbm)
            theta_list.append(float(np.mean(th_vals)))
            phi_list.append(float(np.mean(ph_vals)))
            eirp_list.append(eirp)

    for rx_name in list(sc.receivers.keys()):
        sc.remove(rx_name)

    return (np.array(theta_list), np.array(phi_list), np.array(eirp_list))


# =========================================================================
# 3. HORIZONTAL BEAMWIDTH (H-BW) – "envelope" method
# =========================================================================
def measure_hbw_envelope(sc, tx_name, radius=100.0, tx_power_dbm=53.0,
                          el_span=25.0, az_span=60.0, n_el=30, n_az=40,
                          n_bins=50, center_el_deg=4.0):
    """
    Measures horizontal -3 dB beamwidth. Scans a 2D grid, then for each bin
    of local azimuth takes the maximum EIRP over elevation (envelope). This
    is more robust than a fixed-elevation cut, because the array factor
    introduces vertical ripples.
    """
    tx = sc.transmitters[tx_name]
    tx_ori = np.array(tx.orientation, dtype=np.float64).reshape(-1)
    yaw_deg = np.degrees(to_scalar(tx_ori[0]))

    el_offsets = np.linspace(-el_span, el_span, n_el)
    az_offsets = np.linspace(-az_span, az_span, n_az)

    theta_t, phi_t, eirp = scan_local_angles_and_eirp(
        sc, tx_name, el_offsets, az_offsets,
        center_el_deg=center_el_deg, center_az_deg=yaw_deg,
        radius=radius, tx_power_dbm=tx_power_dbm)

    print(f"  [debug H] valid points: {len(eirp)}, "
          f"phi_t range=[{phi_t.min():.1f},{phi_t.max():.1f}]")

    # Bin by phi_t, take max EIRP in each bin
    bin_edges = np.linspace(phi_t.min(), phi_t.max(), n_bins + 1)
    bin_centers, bin_max_eirp = [], []
    for i in range(n_bins):
        m = (phi_t >= bin_edges[i]) & (phi_t < bin_edges[i + 1])
        if m.sum() > 0:
            bin_centers.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
            bin_max_eirp.append(np.max(eirp[m]))

    bin_centers = np.array(bin_centers)
    bin_max_eirp = np.array(bin_max_eirp)

    peak_idx = int(np.argmax(bin_max_eirp))
    peak_phi = bin_centers[peak_idx]
    thresh = bin_max_eirp[peak_idx] - 3.0

    def interp_crossing(i0, direction):
        i = i0
        while True:
            j = i + direction
            if j < 0 or j >= len(bin_max_eirp):
                return bin_centers[i]
            if bin_max_eirp[j] < thresh:
                frac = (bin_max_eirp[i] - thresh) / (bin_max_eirp[i] - bin_max_eirp[j])
                return bin_centers[i] + frac * (bin_centers[j] - bin_centers[i])
            i = j

    left = interp_crossing(peak_idx, -1)
    right = interp_crossing(peak_idx, +1)
    hbw = right - left

    return hbw, peak_phi, bin_centers, bin_max_eirp


# =========================================================================
# 4. EIRP IN A SINGLE DIRECTION (used for F/B)
# =========================================================================
def measure_eirp_at_direction(sc, tx_name, az_offset_deg, el_deg=None, radius=100.0):
    """
    Measures EIRP in one specific direction. Takes an azimuth offset from
    boresight and an absolute elevation (if None, uses antenna tilt).
    Returns dBm or -inf if no valid path.
    """
    tx = sc.transmitters[tx_name]
    tx_pos = np.array(tx.position, dtype=np.float64).reshape(-1)[:3]
    tx_ori = np.array(tx.orientation, dtype=np.float64).reshape(-1)
    tx_yaw = float(tx_ori[0])
    tx_tilt_rad = float(tx_ori[1])
    p_tx_dbm = float(np.asarray(tx.power_dbm).flatten()[0])

    if el_deg is None:
        el_deg = np.degrees(tx_tilt_rad)  # boresight elevation ≈ tilt

    az_rad = tx_yaw + np.deg2rad(az_offset_deg)
    el_rad = np.radians(-el_deg)

    dx = radius * np.cos(el_rad) * np.cos(az_rad)
    dy = radius * np.cos(el_rad) * np.sin(az_rad)
    dz = radius * np.sin(el_rad)
    rx_pos = [float(tx_pos[0] + dx), float(tx_pos[1] + dy), float(tx_pos[2] + dz)]

    for rx_name in list(sc.receivers.keys()):
        sc.remove(rx_name)
    sc.add(Receiver(name="fbr_probe", position=rx_pos))
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                               horizontal_spacing=0.5, pattern="iso", polarization="cross")

    solver = PathSolver()
    paths = solver(scene=sc, max_depth=0)
    a_real, a_imag = paths.a
    a_complex = np.array(a_real) + 1j * np.array(a_imag)

    tx_idx = list(sc.transmitters.keys()).index(tx_name)
    a_slice = a_complex[0, :, tx_idx, :, :]  # (rx_ant, tx_ant, paths) – single RX

    dist = np.linalg.norm(np.array(rx_pos) - tx_pos)
    lambda_m = config.SPEED_OF_LIGHT_M_S / sc.frequency
    fspl_db = 20.0 * np.log10(4 * np.pi * dist / lambda_m)

    # Same formula: coherent across TX, incoherent across RX and paths
    p_lin = 0.0
    for p in range(a_slice.shape[2]):
        a_path = a_slice[:, :, p]
        a_coherent = np.sum(a_path, axis=1)
        p_lin += np.sum(np.abs(a_coherent) ** 2)
    num_tx_ant = a_slice.shape[1]
    p_lin /= num_tx_ant

    sc.remove("fbr_probe")

    if p_lin <= 0:
        return -np.inf
    return 10 * np.log10(p_lin) + fspl_db + p_tx_dbm


# =========================================================================
# 5. FRONT-TO-BACK RATIO (F/B)
# =========================================================================
def measure_front_to_back_ratio_singleprobe(sc, tx_name, radius=100.0,
                                             back_span_deg=30.0, n_back_points=31,
                                             el_span_deg=60.0, n_el_points=61):
    """
    F/B ratio = EIRP(boresight) - max EIRP in the rear hemisphere. Scans
    azimuth around 180° and elevation around tilt, to avoid missing sidelobes.
    This variant calls measure_eirp_at_direction per point (slower but clearer).

    Datasheet target: FBR_DATASHEET (25 dB).
    """
    print(f"\n{'=' * 60}\nF/B RATIO TEST: {tx_name}\n{'=' * 60}")

    tx = sc.transmitters[tx_name]
    tx_tilt_rad = float(np.array(tx.orientation, dtype=np.float64).reshape(-1)[1])
    boresight_el = np.degrees(tx_tilt_rad)

    eirp_front = measure_eirp_at_direction(sc, tx_name, az_offset_deg=0.0,
                                            el_deg=boresight_el, radius=radius)
    print(f"EIRP boresight (front): {eirp_front} dBm @ el={boresight_el}°")

    back_offsets = 180.0 + np.linspace(-back_span_deg, back_span_deg, n_back_points)
    el_sweep = boresight_el + np.linspace(-el_span_deg, el_span_deg, n_el_points)

    eirp_back_grid = np.full((len(el_sweep), len(back_offsets)), np.nan)
    for i, el in enumerate(el_sweep):
        for j, az in enumerate(back_offsets):
            eirp_back_grid[i, j] = measure_eirp_at_direction(
                sc, tx_name, az_offset_deg=az, el_deg=el, radius=radius)

    flat_idx = np.nanargmax(eirp_back_grid)
    i_max, j_max = np.unravel_index(flat_idx, eirp_back_grid.shape)
    eirp_back_max = eirp_back_grid[i_max, j_max]
    worst_az = back_offsets[j_max]
    worst_el = el_sweep[i_max]

    fbr_measured = eirp_front - eirp_back_max

    print(f"Max EIRP in rear hemisphere: {eirp_back_max} dBm "
          f"@ azimuth_offset={worst_az:.1f}°, elevation={worst_el}°")
    print(f"\n>>> Measured F/B ratio: {fbr_measured} dB  (target: {config.FBR_DATASHEET} dB)")
    print(f">>> Deviation from datasheet: {fbr_measured - config.FBR_DATASHEET} dB")

    return {"eirp_front": eirp_front, "eirp_back_max": eirp_back_max,
            "fbr_measured": fbr_measured, "worst_az": worst_az, "worst_el": worst_el,
            "back_offsets": back_offsets, "el_sweep": el_sweep,
            "eirp_back_grid": eirp_back_grid,
            "eirp_back_arr": np.nanmax(eirp_back_grid, axis=0)}


def measure_front_to_back_ratio_batch(sc, tx_name, radius=100.0,
                                       back_span_deg=30.0, n_back_points=31,
                                       el_span_deg=60.0, n_el_points=61):
    """
    Same as singleprobe, but optimised: all rear probes are added at once
    and PathSolver is called only once. Recommended for larger grids.
    """
    print(f"\n{'=' * 60}\nF/B RATIO TEST (batch): {tx_name}\n{'=' * 60}")

    tx = sc.transmitters[tx_name]
    tx_pos = np.array(tx.position, dtype=np.float64).reshape(-1)[:3]
    tx_ori = np.array(tx.orientation, dtype=np.float64).reshape(-1)
    tx_yaw = float(tx_ori[0])
    tx_tilt_rad = float(tx_ori[1])
    p_tx_dbm = float(np.asarray(tx.power_dbm).flatten()[0])
    boresight_el = np.degrees(tx_tilt_rad)

    # Front measurement – single probe
    eirp_front = measure_eirp_at_direction(sc, tx_name, az_offset_deg=0.0,
                                            el_deg=boresight_el, radius=radius)
    print(f"EIRP boresight (front): {eirp_front} dBm @ el={boresight_el}°")

    # Generate all rear hemisphere positions
    back_offsets = 180.0 + np.linspace(-back_span_deg, back_span_deg, n_back_points)
    el_sweep = boresight_el + np.linspace(-el_span_deg, el_span_deg, n_el_points)

    for rx_name in list(sc.receivers.keys()):
        sc.remove(rx_name)

    rx_names = []
    for i, el in enumerate(el_sweep):
        for j, az in enumerate(back_offsets):
            az_rad = tx_yaw + np.deg2rad(az)
            el_rad = np.radians(-el)
            dx = radius * np.cos(el_rad) * np.cos(az_rad)
            dy = radius * np.cos(el_rad) * np.sin(az_rad)
            dz = radius * np.sin(el_rad)
            rx_pos = [float(tx_pos[0] + dx), float(tx_pos[1] + dy), float(tx_pos[2] + dz)]
            rx_name = f"back_probe_{i}_{j}"
            sc.add(Receiver(name=rx_name, position=rx_pos))
            rx_names.append(rx_name)

    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                               horizontal_spacing=0.5, pattern="iso", polarization="cross")

    # Single PathSolver call for all probes
    solver = PathSolver()
    paths = solver(scene=sc, max_depth=0)
    a_real, a_imag = paths.a
    a_complex = np.array(a_real) + 1j * np.array(a_imag)

    tx_idx = list(sc.transmitters.keys()).index(tx_name)
    eirp_back_grid = np.full((len(el_sweep), len(back_offsets)), np.nan)
    lambda_m = config.SPEED_OF_LIGHT_M_S / sc.frequency
    fspl_db = 20.0 * np.log10(4 * np.pi * radius / lambda_m)  # same distance for all

    for k, rx_name in enumerate(rx_names):
        rx_idx = list(sc.receivers.keys()).index(rx_name)
        a_slice = a_complex[rx_idx, :, tx_idx, :, :]  # (rx_ant, tx_ant, paths)
        num_tx_ant = a_slice.shape[1]
        p_lin = 0.0
        for p in range(a_slice.shape[2]):
            a_path = a_slice[:, :, p]
            a_coherent = np.sum(a_path, axis=1)
            p_lin += np.sum(np.abs(a_coherent) ** 2)
        p_lin /= num_tx_ant
        eirp_val = 10 * np.log10(p_lin) + fspl_db + p_tx_dbm if p_lin > 0 else np.nan

        i = k // len(back_offsets)
        j = k % len(back_offsets)
        eirp_back_grid[i, j] = eirp_val

    for rx_name in rx_names:
        sc.remove(rx_name)

    flat_idx = np.nanargmax(eirp_back_grid)
    i_max, j_max = np.unravel_index(flat_idx, eirp_back_grid.shape)
    eirp_back_max = eirp_back_grid[i_max, j_max]
    worst_az = back_offsets[j_max]
    worst_el = el_sweep[i_max]

    fbr_measured = eirp_front - eirp_back_max

    print(f"Max EIRP in rear hemisphere: {eirp_back_max} dBm "
          f"@ azimuth_offset={worst_az:.1f}°, elevation={worst_el}°")
    print(f"\n>>> Measured F/B ratio: {fbr_measured} dB  (target: {config.FBR_DATASHEET} dB)")
    print(f">>> Deviation from datasheet: {fbr_measured - config.FBR_DATASHEET} dB")

    return {"eirp_front": eirp_front, "eirp_back_max": eirp_back_max,
            "fbr_measured": fbr_measured, "worst_az": worst_az, "worst_el": worst_el,
            "back_offsets": back_offsets, "el_sweep": el_sweep,
            "eirp_back_grid": eirp_back_grid}


# =========================================================================
# 6. VERTICAL SIDE-LOBE LEVEL (SLL)
# =========================================================================
def measure_side_lobe_level_vertical(sc, tx_name, radius=100.0, tx_power_dbm=53.0,
                                      span_deg=60.0, n_points=400):
    """
    Measures vertical SLL – difference between main lobe peak and the first
    side-lobe peak. Algorithm: from the main peak, go both ways, follow the
    descent to the first null, then the ascent to the next maximum.
    Takes the higher of the two side lobes (worst case).

    Datasheet target: SLL_DATASHEET (13 dB).
    """
    tx = sc.transmitters[tx_name]
    tx.power_dbm = tx_power_dbm
    p_tx_dbm = to_scalar(tx.power_dbm)
    tx_pos = np.array(tx.position, dtype=np.float64).reshape(-1)[:3]
    tx_ori = np.array(tx.orientation, dtype=np.float64).reshape(-1)
    tx_yaw = to_scalar(tx_ori[0])

    for rx_name in list(sc.receivers.keys()):
        sc.remove(rx_name)

    elevations = np.linspace(-span_deg, span_deg, n_points)
    positions = []
    for el in elevations:
        el_rad = np.radians(-el)
        dx = radius * np.cos(el_rad) * np.cos(tx_yaw)
        dy = radius * np.cos(el_rad) * np.sin(tx_yaw)
        dz = radius * np.sin(el_rad)
        pos = [float(tx_pos[0] + dx), float(tx_pos[1] + dy), float(tx_pos[2] + dz)]
        positions.append(pos)
        sc.add(Receiver(name=f"sll_probe_{len(positions)}", position=pos))

    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                               horizontal_spacing=0.5, pattern="iso", polarization="cross")
    solver = PathSolver()
    paths = solver(scene=sc, max_depth=0)
    a_np = np.array(paths.a)

    n = len(positions)
    rx_axis = next((i for i, s in enumerate(a_np.shape) if s == n), 0)
    tx_ant_axis_original = 4
    num_tx_ant = a_np.shape[tx_ant_axis_original]

    eirp = []
    for k in range(n):
        slices = [slice(None)] * len(a_np.shape)
        slices[rx_axis] = k
        a_rx = a_np[tuple(slices)]
        tx_ant_axis = tx_ant_axis_original if rx_axis > tx_ant_axis_original \
            else tx_ant_axis_original - 1
        a_coherent = np.sum(a_rx, axis=tx_ant_axis)
        power_lin = np.sum(np.abs(a_coherent) ** 2) / num_tx_ant
        eirp_val = 10 * np.log10(power_lin) + p_tx_dbm if power_lin > 0 else np.nan
        eirp.append(eirp_val)

    for rx_name in list(sc.receivers.keys()):
        sc.remove(rx_name)

    eirp = np.array(eirp)

    main_idx = int(np.nanargmax(eirp))
    main_peak_db = eirp[main_idx]
    main_peak_el = elevations[main_idx]

    def find_first_null_and_sidelobe_peak(direction):
        """
        From main peak in direction `direction` (1 or -1):
        first descending edge to the first null, then ascending to side peak.
        """
        i = main_idx
        # Phase 1: descend along the main lobe
        while True:
            j = i + direction
            if j < 0 or j >= len(eirp) or np.isnan(eirp[j]):
                return None, None
            if eirp[j] > eirp[i]:
                break
            i = j
        null_idx = i
        # Phase 2: ascend to side-lobe peak
        while True:
            j = i + direction
            if j < 0 or j >= len(eirp) or np.isnan(eirp[j]):
                return null_idx, i
            if eirp[j] < eirp[i]:
                return null_idx, i
            i = j

    null_left, peak_left = find_first_null_and_sidelobe_peak(-1)
    null_right, peak_right = find_first_null_and_sidelobe_peak(+1)

    candidates = []
    if peak_left is not None:
        candidates.append(("left", eirp[peak_left], elevations[peak_left]))
    if peak_right is not None:
        candidates.append(("right", eirp[peak_right], elevations[peak_right]))

    if not candidates:
        raise RuntimeError("No side lobe found within scan range – increase span_deg.")

    # Take the higher side lobe (worst case)
    side_dir, side_peak_db, side_peak_el = max(candidates, key=lambda c: c[1])

    sll_measured = main_peak_db - side_peak_db

    print(f"Main lobe: {main_peak_db} dBm @ elevation={main_peak_el}°")
    for d, val, el in candidates:
        print(f"First side lobe ({d}): {val} dBm @ elevation={el}°")
    print(f"Dominant side lobe ({side_dir}): {side_peak_db} dBm")
    print(f"\n>>> Measured Vertical SLL: {sll_measured} dB  (target: {config.SLL_DATASHEET:.1f} dB)")
    print(f">>> Deviation from datasheet: {sll_measured - config.SLL_DATASHEET} dB")

    return {"main_peak_db": main_peak_db, "main_peak_el": main_peak_el,
            "side_peak_db": side_peak_db, "sll_measured": sll_measured,
            "elevations": elevations, "eirp": eirp}

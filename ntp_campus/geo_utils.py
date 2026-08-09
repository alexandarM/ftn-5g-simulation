"""
geo_utils.py
=============

Georeferencing layer for the NTP Campus 5G Coverage Simulation.

Maps between the Sionna RT local coordinate system (X = east, Y = north) and
real WGS84 lat/lon coordinates of the FTN campus. Enables overlay of RadioMap
output onto satellite/OSM backgrounds, similar to the BostonTwin project
(which uses geopandas + contextily).

Works directly with the `sionna.rt.RadioMap` object returned by
`radio_map.compute_radio_map()` (PlanarRadioMap subclass):
    - cell_centers : mi.TensorXf, shape [cells_y, cells_x, 3]
    - path_gain / rss : mi.TensorXf, shape [num_tx, cells_y, cells_x], LINEAR scale

Dependencies (see requirements.txt):
    pyproj
    contextily

One‑time setup – CAMPUS_REF_LAT / CAMPUS_REF_LON / CAMPUS_REF_ROTATION_DEG
are defined in config.py. The defaults there point at the FTN campus
(Trg Dositeja Obradovića 6, Novi Sad) as a reasonable starting point, but
the EXACT value that will make the overlay line up with real geometry
depends on wherever (0, 0) was placed during the BLOSM import in Blender.

How to find/refine CAMPUS_REF_LAT/LON:
    Find the exact point on Google Maps that was used as (0,0) in Blender
    during the BLOSM import (usually the scene centre or the reference point
    that BLOSM placed at import). Right‑click -> copy coordinates, then
    update config.py. You can sanity-check the result with
    `plot_coverage_geo()`: if transmitter markers land on the correct
    physical mast locations on the satellite image, the reference point
    (and rotation) are correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config


# =========================================================================
# 1. GEOREFERENCE – scene (x, y) <-> (lat, lon) conversion
# =========================================================================
@dataclass
class GeoReference:
    """Reference point linking local (x, y) [m] coordinates from the
    Sionna scene (X = east, Y = north) to WGS84 lat/lon.
    """

    ref_lat: float
    ref_lon: float
    utm_epsg: int = 32634          # UTM zone 34N – covers Novi Sad / FTN campus
    rotation_deg: float = 0.0      # scene rotation relative to true north (CW+)

    _to_utm: "object" = field(init=False, repr=False)
    _to_wgs84: "object" = field(init=False, repr=False)
    _to_merc: "object" = field(init=False, repr=False)
    ref_easting: float = field(init=False, repr=False)
    ref_northing: float = field(init=False, repr=False)

    def __post_init__(self):
        from pyproj import Transformer
        self._to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
        self._to_wgs84 = Transformer.from_crs(f"EPSG:{self.utm_epsg}", "EPSG:4326", always_xy=True)
        self._to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.ref_easting, self.ref_northing = self._to_utm.transform(self.ref_lon, self.ref_lat)

    def local_to_latlon(self, x, y):
        """Local scene (x, y) [m] -> (lat, lon). Works with scalars and
        numpy arrays (vectorised, via pyproj's vectorised `transform`)."""
        theta = math.radians(self.rotation_deg)
        d_east = x * math.cos(theta) - y * math.sin(theta)
        d_north = x * math.sin(theta) + y * math.cos(theta)
        easting = self.ref_easting + d_east
        northing = self.ref_northing + d_north
        lon, lat = self._to_wgs84.transform(easting, northing)
        return lat, lon

    def latlon_to_local(self, lat, lon):
        """(lat, lon) -> local scene (x, y) [m]."""
        easting, northing = self._to_utm.transform(lon, lat)
        d_east = easting - self.ref_easting
        d_north = northing - self.ref_northing
        theta = math.radians(-self.rotation_deg)
        x = d_east * math.cos(theta) - d_north * math.sin(theta)
        y = d_east * math.sin(theta) + d_north * math.cos(theta)
        return x, y

    def local_to_webmercator(self, x, y):
        """Local (x, y) -> EPSG:3857, the format expected by contextily.
        Works with scalars OR numpy arrays of any shape (e.g. a full 2D
        grid of cell centres) -- pyproj's `Transformer.transform` is
        vectorised, so passing a whole grid at once is both correct AND
        fast (no Python-level loop needed)."""
        lat, lon = self.local_to_latlon(x, y)
        return self._to_merc.transform(lon, lat)


def get_campus_geo_reference() -> GeoReference:
    """Builds a `GeoReference` from constants defined in `config.py`
    (CAMPUS_REF_LAT, CAMPUS_REF_LON, CAMPUS_REF_ROTATION_DEG).
    """
    return GeoReference(
        ref_lat=config.CAMPUS_REF_LAT,
        ref_lon=config.CAMPUS_REF_LON,
        rotation_deg=getattr(config, "CAMPUS_REF_ROTATION_DEG", 0.0),
    )


# =========================================================================
# 2. EXTRACT GRID AND METRICS FROM THE sionna.rt.RadioMap OBJECT
# =========================================================================
def extract_radio_map_grid(radio_map, metric="path_gain", tx=None, to_db=True):
    """Extracts the (x, y) grid and metric values from the `RadioMap` object
    returned by `radio_map.compute_radio_map()`.

    Parameters
    ----------
    radio_map : sionna.rt.RadioMap (PlanarRadioMap)
        Output of `compute_radio_map()`.
    metric : {"path_gain", "rss", "sinr"}
        Same metric names used in `show_all_radio_map_metrics`.
    tx : int or None
        If given, uses only that transmitter (index order matches
        `scene.transmitters`, same as `rm_tx` in `render_coverage`).
        If None, combines all transmitters by taking the BEST (max) signal
        per cell – a "best server" view, identical to Sionna's internal
        behaviour when `radio_map.show(metric=...)` is called without `tx`.
    to_db : bool
        Converts from linear scale to dB (10*log10). `sinr` is already in
        dB in some Sionna versions – if you see double‑converted values
        (e.g., -300 dB), call with `to_db=False`.

    Returns
    -------
    x_centers, y_centers : np.ndarray, shape (cells_x,), (cells_y,)
        Cell centre coordinates (for extent calculation).
    values : np.ndarray, shape (cells_y, cells_x)
        Metric values (in dB if to_db=True).
    """
    centers = np.array(radio_map.cell_centers)  # [cells_y, cells_x, 3]
    x_centers = centers[0, :, 0]                # X (east) varies by column
    y_centers = centers[:, 0, 1]                # Y (north) varies by row

    data = np.array(getattr(radio_map, metric))  # [num_tx, cells_y, cells_x]

    if tx is not None:
        values = data[tx]
    else:
        values = np.max(data, axis=0)  # best-server combination

    if to_db:
        with np.errstate(divide="ignore"):
            values = 10.0 * np.log10(np.maximum(values, 1e-15))

    return x_centers, y_centers, values


def get_transmitter_positions(scene):
    """Returns {tx_name: (x, y)} for all current transmitters in the scene
    (directly from `scene.transmitters`, same source as used by
    `transmitters.print_transmitter_powers`)."""
    positions = {}
    for name, tx in scene.transmitters.items():
        pos = np.array(tx.position).flatten()
        positions[name] = (float(pos[0]), float(pos[1]))
    return positions


# =========================================================================
# 3. OVERLAY OVER A REAL SATELLITE/OSM BACKGROUND
# =========================================================================
def plot_coverage_geo(
    scene,
    radio_map,
    geo_ref: GeoReference = None,
    metric: str = "rss",
    tx: int | None = None,
    cmap: str = "plasma", # jet, inferno,coolwarm
    alpha: float = 0.5,
    vmin: float | None = None,
    vmax: float | None = None,
    basemap_source=None,
    zoom: int | str = "auto",
    title: str | None = None,
    save_path: str | None = None,
):
    """Renders `radio_map` (output of `compute_radio_map`) over a real
    satellite/OSM background of the FTN campus, with markers for all currently
    active transmitters in the scene.

    Replacement/enhancement for `radio_map.render_coverage()` 2D preview –
    use it the same way, but also pass `scene` and `geo_ref`:

        from ntp_campus.geo_utils import get_campus_geo_reference, plot_coverage_geo
        geo_ref = get_campus_geo_reference()
        plot_coverage_geo(scene, radio_map, geo_ref, metric="path_gain",
                           save_path="figures/coverage_geo_overlay.png")

    """
    import matplotlib.pyplot as plt
    import contextily as cx

    if geo_ref is None:
        geo_ref = get_campus_geo_reference()

    x_centers, y_centers, values = extract_radio_map_grid(radio_map, metric=metric, tx=tx)

    # Full 2D grid of LOCAL cell-centre coordinates (not just the 4 corners),
    # then transform every point individually to Web Mercator. This is what
    # makes the overlay correct for ANY geo_ref.rotation_deg (see docstring).
    X, Y = np.meshgrid(x_centers, y_centers)          # each shape (cells_y, cells_x)
    mx, my = geo_ref.local_to_webmercator(X, Y)        # vectorised pyproj transform

    fig, ax = plt.subplots(figsize=(11, 11))
    # shading="auto": mx/my/values all share shape (cells_y, cells_x), so
    # matplotlib treats mx/my as cell CENTRES (not vertices) automatically.
    im = ax.pcolormesh(mx, my, values, cmap=cmap, alpha=alpha,
                        vmin=vmin, vmax=vmax, shading="auto", zorder=2)

    tx_positions = get_transmitter_positions(scene)
    for name, (tx_x, tx_y) in tx_positions.items():
        tmx, tmy = geo_ref.local_to_webmercator(tx_x, tx_y)
        ax.scatter(tmx, tmy, marker="^", s=160, c="white", edgecolors="black",
                   linewidths=1.5, zorder=4)
        ax.annotate(name, (tmx, tmy), textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.65), zorder=5)

    # Axis limits / basemap extent: use the actual (rotation-aware) min/max
    # of the transformed grid + tx markers, with a small margin, instead of
    # the old corner-only bounding box (which is now only used implicitly
    # here, correctly, since mx/my already include every cell).
    all_mx = np.concatenate([mx.ravel(), [p[0] for p in
                             (geo_ref.local_to_webmercator(x, y) for x, y in tx_positions.values())] or mx.ravel()[:1]])
    all_my = np.concatenate([my.ravel(), [p[1] for p in
                             (geo_ref.local_to_webmercator(x, y) for x, y in tx_positions.values())] or my.ravel()[:1]])
    margin_x = 0.05 * (all_mx.max() - all_mx.min() + 1e-6)
    margin_y = 0.05 * (all_my.max() - all_my.min() + 1e-6)
    ax.set_xlim(all_mx.min() - margin_x, all_mx.max() + margin_x)
    ax.set_ylim(all_my.min() - margin_y, all_my.max() + margin_y)
    ax.set_axis_off()
    ax.set_title(title or f"Coverage ({metric}) over FTN campus")
    fig.colorbar(im, ax=ax, shrink=0.75,
                 label=f"{metric.upper()} [dB]" if metric != "sinr" else "SINR [dB]")

    source = basemap_source or cx.providers.Esri.WorldImagery
    cx.add_basemap(ax, crs="EPSG:3857", source=source, zoom=zoom, zorder=1)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[OK] Saved: {save_path}")

    return fig, ax


if __name__ == "__main__":
    # Quick round‑trip test of coordinate conversion (does not require Sionna/scene).
    ref = GeoReference(ref_lat=45.2463017, ref_lon=19.8513561, rotation_deg=-3.1056)
    lat, lon = ref.local_to_latlon(50.0, 30.0)
    x_back, y_back = ref.latlon_to_local(lat, lon)
    print(f"local(50,30) -> latlon({lat:.6f}, {lon:.6f}) -> local({x_back:.3f}, {y_back:.3f})")

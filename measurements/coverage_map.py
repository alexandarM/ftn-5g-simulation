"""Build a coverage map with a heat map from a drive-test CSV.

    python coverage_map.py logs/drive_20260918_141203.csv
    python coverage_map.py --metric sinr_db -o sinr_map.html

Output is a self-contained HTML file: a weighted heat map plus one point per
measurement, coloured on the same scale, with the full reading in the popup.
"""

import argparse
import csv
import glob
import os
import sys
import webbrowser

import folium
from folium.plugins import HeatMap

# Each metric carries the scale ends used for colouring: "worst" is the value
# that reads as no usable coverage, "best" the value that reads as excellent.
# For rtt_ms the pair is inverted, which flips the ramp automatically.
METRICS = {
    "rsrp_dbm": {"label": "RSRP", "unit": "dBm", "worst": -120.0, "best": -70.0},
    "rsrq_db": {"label": "RSRQ", "unit": "dB", "worst": -20.0, "best": -3.0},
    "sinr_db": {"label": "SINR / SNR", "unit": "dB", "worst": -5.0, "best": 30.0},
    "rssi_dbm": {"label": "RSSI", "unit": "dBm", "worst": -105.0, "best": -55.0},
    "cqi": {"label": "CQI", "unit": "", "worst": 0.0, "best": 15.0},
    "dl_mbps": {"label": "Downlink", "unit": "Mbps", "worst": 0.0, "best": 200.0},
    "ul_mbps": {"label": "Uplink", "unit": "Mbps", "worst": 0.0, "best": 50.0},
    "rtt_ms": {"label": "RTT", "unit": "ms", "worst": 300.0, "best": 10.0},
}

# Sequential blue ramp, light (weakest) to dark (strongest). Five ordered steps
# taken from the reference palette, starting at the step that still clears the
# contrast floor on a light basemap.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
BINS = len(RAMP)

# Chart chrome, light tokens: the basemap is always light, so the legend keeps
# the light surface and ink and matches the marker colours exactly.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
HAIRLINE = "rgba(11,11,11,0.10)"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

POPUP_FIELDS = [
    ("timestamp_local", "Time", ""),
    ("rat", "RAT", ""),
    ("operator", "Operator", ""),
    ("cell_id", "Cell ID", ""),
    ("pci", "PCI", ""),
    ("band", "Band", ""),
    ("rsrp_dbm", "RSRP", "dBm"),
    ("rsrq_db", "RSRQ", "dB"),
    ("sinr_db", "SINR", "dB"),
    ("rssi_dbm", "RSSI", "dBm"),
    ("cqi", "CQI", ""),
    ("rtt_ms", "RTT", "ms"),
    ("dl_mbps", "Downlink", "Mbps"),
    ("ul_mbps", "Uplink", "Mbps"),
    ("satellites", "Satellites", ""),
]


def fmt(value):
    """Trim a float to the shortest sensible text."""
    if abs(value - round(value)) < 0.05:
        return "%d" % round(value)
    return "%.1f" % value


def normalise(value, metric):
    """Map a metric value onto 0 (worst) .. 1 (best), clamped."""
    span = metric["best"] - metric["worst"]
    if span == 0:
        return 0.0
    return max(0.0, min(1.0, (value - metric["worst"]) / span))


def bin_index(value, metric):
    return min(BINS - 1, int(normalise(value, metric) * BINS))


def bin_labels(metric):
    """Value range text for each bin, worst first."""
    step = (metric["best"] - metric["worst"]) / BINS
    labels = []
    for index in range(BINS):
        low = metric["worst"] + index * step
        high = low + step
        if index == 0:
            labels.append("%s %s" % ("<" if step > 0 else ">", fmt(high)))
        elif index == BINS - 1:
            labels.append("%s %s" % (">=" if step > 0 else "<=", fmt(low)))
        else:
            labels.append("%s to %s" % (fmt(min(low, high)), fmt(max(low, high))))
    return labels


def read_samples(path, metric_key):
    """Rows that have both a position and a value for the chosen metric."""
    samples = []
    total = 0
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                value = float(row[metric_key])
            except (KeyError, TypeError, ValueError):
                continue
            row["_lat"] = lat
            row["_lon"] = lon
            row["_value"] = value
            samples.append(row)
    return samples, total


def popup_html(row):
    rows = []
    for key, label, unit in POPUP_FIELDS:
        value = (row.get(key) or "").strip()
        if not value:
            continue
        rows.append(
            '<tr><td style="padding:1px 10px 1px 0;color:%s">%s</td>'
            '<td style="padding:1px 0;color:%s;font-variant-numeric:tabular-nums">'
            "%s %s</td></tr>" % (INK_MUTED, label, INK, value, unit)
        )
    return ('<div style="font:12px/1.5 %s;min-width:190px">'
            "<table>%s</table></div>" % (FONT, "".join(rows)))


def legend_html(metric, counts, point_count, path):
    labels = bin_labels(metric)
    unit = (" " + metric["unit"]) if metric["unit"] else ""
    rows = []
    for index in reversed(range(BINS)):
        rows.append(
            '<div style="display:flex;align-items:center;gap:8px;padding:2px 0">'
            '<span style="width:14px;height:14px;border-radius:3px;'
            'background:%s;box-shadow:0 0 0 1px %s;flex:none"></span>'
            '<span style="flex:1;color:%s">%s%s</span>'
            '<span style="color:%s;font-variant-numeric:tabular-nums">%d</span>'
            "</div>" % (RAMP[index], HAIRLINE, INK, labels[index], unit,
                        INK_MUTED, counts[index])
        )
    return (
        '<div style="position:fixed;bottom:22px;left:12px;z-index:9999;'
        "background:%s;border:1px solid %s;border-radius:8px;padding:10px 12px;"
        'font:12px/1.4 %s;box-shadow:0 1px 4px rgba(0,0,0,.12)">'
        '<div style="font-weight:600;color:%s;margin-bottom:6px">%s coverage</div>'
        "%s"
        '<div style="margin-top:6px;color:%s;max-width:210px">%d points. Darker '
        "is stronger; the count column shows how many points fall in each band. "
        "Exact values: popups, or the table in %s.</div></div>"
        % (SURFACE, HAIRLINE, FONT, INK, metric["label"], "".join(rows),
           INK_MUTED, point_count, os.path.basename(path))
    )


def build_map(samples, metric_key, path, max_markers, heat_radius):
    metric = METRICS[metric_key]
    lats = [s["_lat"] for s in samples]
    lons = [s["_lon"] for s in samples]
    fmap = folium.Map(
        location=[sum(lats) / len(lats), sum(lons) / len(lons)],
        zoom_start=14,
        tiles=None,
    )
    # A recessive grey canvas keeps the data on top; both basemaps are
    # key-free (Carto's public tiles are watermarked now).
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/"
              "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, NGA, USGS",
        name="Grey canvas",
    ).add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="Street map", show=False).add_to(fmap)

    counts = [0] * BINS
    heat_points = []
    for sample in samples:
        index = bin_index(sample["_value"], metric)
        counts[index] += 1
        # Floor the weight so that the weakest locations still show up on the
        # heat map instead of fading into the basemap.
        weight = 0.1 + 0.9 * normalise(sample["_value"], metric)
        heat_points.append([sample["_lat"], sample["_lon"], weight])

    HeatMap(
        heat_points,
        name="Heat map (%s)" % metric["label"],
        radius=heat_radius,
        blur=int(heat_radius * 0.6),
        min_opacity=0.25,
        gradient={0.0: "#cde2fb", 0.25: "#86b6ef", 0.5: "#3987e5",
                  0.75: "#1c5cab", 1.0: "#104281"},
    ).add_to(fmap)

    route = folium.FeatureGroup(name="Route", show=False)
    folium.PolyLine(list(zip(lats, lons)), color=INK_MUTED, weight=2,
                    opacity=0.5).add_to(route)
    route.add_to(fmap)

    points = folium.FeatureGroup(name="Measurement points")
    step = max(1, len(samples) // max_markers) if max_markers else 1
    unit = (" " + metric["unit"]) if metric["unit"] else ""
    for sample in samples[::step]:
        folium.CircleMarker(
            location=[sample["_lat"], sample["_lon"]],
            radius=5,
            color=SURFACE,            # 2px surface ring keeps overlaps readable
            weight=2,
            fill=True,
            fill_color=RAMP[bin_index(sample["_value"], metric)],
            fill_opacity=0.95,
            tooltip="%s %s%s" % (metric["label"], fmt(sample["_value"]), unit),
            popup=folium.Popup(popup_html(sample), max_width=320),
        ).add_to(points)
    points.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    fmap.get_root().html.add_child(
        folium.Element(legend_html(metric, counts, len(samples), path)))
    return fmap, step


def newest_log():
    files = sorted(glob.glob(os.path.join("logs", "*.csv")), key=os.path.getmtime)
    return files[-1] if files else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", help="drive-test CSV (default: newest in logs/)")
    parser.add_argument("--metric", default="rsrp_dbm", choices=sorted(METRICS),
                        help="metric to colour by (default rsrp_dbm)")
    parser.add_argument("-o", "--output", help="output HTML (default <csv>_<metric>.html)")
    parser.add_argument("--max-markers", type=int, default=2000,
                        help="subsample point markers above this count (0 = all)")
    parser.add_argument("--heat-radius", type=int, default=18,
                        help="heat map point radius in pixels")
    parser.add_argument("--open", action="store_true", help="open the map in a browser")
    args = parser.parse_args(argv)

    path = args.csv or newest_log()
    if not path:
        print("Error: no CSV given and none found in logs/.")
        return 1
    if not os.path.isfile(path):
        print("Error: %s not found." % path)
        return 1

    samples, total = read_samples(path, args.metric)
    if not samples:
        print("Error: no rows in %s have both a GPS position and %s."
              % (path, args.metric))
        return 1

    fmap, step = build_map(samples, args.metric, path, args.max_markers,
                           args.heat_radius)
    output = args.output or "%s_%s.html" % (os.path.splitext(path)[0], args.metric)
    fmap.save(output)

    print("Read %d rows, mapped %d with a position and a %s value."
          % (total, len(samples), args.metric))
    if step > 1:
        print("Markers subsampled 1 in %d (%d shown); the heat map uses every point."
              % (step, len(samples[::step])))
    print("Map written to %s" % output)
    if args.open:
        webbrowser.open("file://" + os.path.abspath(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())

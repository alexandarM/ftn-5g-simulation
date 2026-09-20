import argparse
import csv
import os
import sys
import folium
from folium.plugins import HeatMap

# Metrics configuration with Viridis color scales (Figure 6.12 from Sionne)
METRICS = {
    "rsrp_dbm": {
        "label": "RSRP (dBm)",
        "min_val": -125.0,  # Sensitivity threshold of the modem (dark purple)
        "max_val": -75.0,   # Excellent signal in LOS area (yellow)
        "unit": "dBm",
    },
    "sinr_db": {
        "label": "SINR (dB)",
        "min_val": -5.0,
        "max_val": 25.0,
        "unit": "dB",
    },
    "rsrq_db": {
        "label": "RSRQ (dB)",
        "min_val": -19.5,
        "max_val": -3.0,
        "unit": "dB",
    },
    "rssi_dbm": {
        "label": "RSSI (dBm)",
        "min_val": -100.0,
        "max_val": -50.0,
        "unit": "dBm",
    },
}

# Colors for Sionna sectors (Figure 6.13: TX 0 = Purple, TX 1 = Mint green, TX 2 = Red)
SECTOR_COLORS = {
    0: "#800080",  # TX 0 (Purple)
    1: "#00e676",  # TX 1 (Mint green)
    2: "#d50000",  # TX 2 (Red)
}


def get_viridis_color(val, min_v, max_v):
    """Returns the hexadecimal color from the Viridis scale for discrete points."""
    if val is None:
        return "#808080"
    norm = max(0.0, min(1.0, (val - min_v) / (max_v - min_v)))
    if norm < 0.25:
        return "#440154"  # Dark purple
    elif norm < 0.50:
        return "#3b528b"  # Blue
    elif norm < 0.75:
        return "#21918c"  # Turquoise
    elif norm < 0.90:
        return "#5ec962"  # Light green
    else:
        return "#fde725"  # Yellow


def main():
    parser = argparse.ArgumentParser(description="Generation of Sionna-compatible coverage maps.")
    parser.add_argument("csv_path", help="Path to the recorded drive-test CSV file")
    parser.add_argument(
        "--metric",
        default="rsrp_dbm",
        choices=list(METRICS.keys()),
        help="Radio metric for display (default: rsrp_dbm)",
    )
    parser.add_argument("--radius", type=int, default=12, help="Radius of the heat map (default: 12)")
    parser.add_argument("--blur", type=int, default=6, help="Blur level of the heat map (default: 6)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: File '{args.csv_path}' does not exist.")
        sys.exit(1)

    cfg = METRICS[args.metric]
    rows = []
    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    valid_points = []
    heat_points = []
    route_coords = []

    for r in rows:
        lat_str = r.get("lat", "")
        lon_str = r.get("lon", "")
        val_str = r.get(args.metric, "")

        if not lat_str or not lon_str:
            continue

        try:
            lat = float(lat_str)
            lon = float(lon_str)
            val = float(val_str) if val_str not in ("", "None", None) else None
        except ValueError:
            continue

        pci_str = r.get("pci", "")
        sector_id = None
        try:
            if pci_str and pci_str not in ("", "None", None):
                sector_id = int(pci_str) % 3  # Sector association: PCI mod 3
        except ValueError:
            sector_id = None

        pt = {
            "lat": lat,
            "lon": lon,
            "val": val,
            "pci": pci_str,
            "sector_id": sector_id,
            "cell_id": r.get("cell_id", "-"),
            "band": r.get("band", "-"),
            "time": r.get("timestamp_utc", "-"),
        }
        valid_points.append(pt)
        route_coords.append((lat, lon))

        if val is not None:
            # Normalizacija težine za toplotnu mapu (0.0 do 1.0)
            norm_weight = max(0.0, min(1.0, (val - cfg["min_val"]) / (cfg["max_val"] - cfg["min_val"])))
            heat_points.append([lat, lon, norm_weight])

    if not valid_points:
        print(f"No valid points with GPS coordinates in file {args.csv_path}.")
        sys.exit(1)

    center_lat = sum(p["lat"] for p in valid_points) / len(valid_points)
    center_lon = sum(p["lon"] for p in valid_points) / len(valid_points)

    # Main map with deep zoom allowed up to level 20
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        max_zoom=20,
        tiles=None,
    )

    # 1. Neutral gray base map (Sionna style)
    # max_native_zoom=16 prevents the "Map data not yet available" issue
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, NGA, USGS",
        name="Grey canvas (Sionna)",
        max_zoom=20,
        max_native_zoom=16,
    ).add_to(fmap)

    # 2. Street map (Esri World Street - does not block local opening)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Street map",
        max_zoom=19,
        show=False,
    ).add_to(fmap)

    # 3. Satellite imagery (ideal for analyzing buildings and obstacles on campus)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite (Buildings and terrain)",
        max_zoom=19,
        show=False,
    ).add_to(fmap)

    # 1. HeatMap layer (Viridis scale based on Figure 6.12)
    HeatMap(
        heat_points,
        name=f"Heat map ({cfg['label']}) [Sionna Viridis]",
        radius=args.radius,
        blur=args.blur,
        min_opacity=0.45,
        gradient={
            0.0: "#440154",   # Dark purple (very weak signal / building shadow)
            0.25: "#3b528b",  # Dark blue
            0.50: "#21918c",  # Turquoise
            0.75: "#5ec962",  # Light green
            1.0: "#fde725",   # Yellow (very strong signal / LOS)
        },
    ).add_to(fmap)

    # 2. Association layer for sectors (PCI mod 3 - Figure 6.13 from Sionna)
    sector_group = folium.FeatureGroup(name="Sionna TX Sectors (PCI mod 3)", show=False)
    for p in valid_points:
        sec = p["sector_id"]
        sec_color = SECTOR_COLORS.get(sec, "#808080")
        popup_txt = (
            f"<b>Time:</b> {p['time']}<br>"
            f"<b>PCI:</b> {p['pci']}<br>"
            f"<b>Sector (PCI % 3):</b> TX {sec if sec is not None else 'N/A'}<br>"
            f"<b>Cell ID:</b> {p['cell_id']}<br>"
            f"<b>{cfg['label']}:</b> {p['val']} {cfg['unit']}"
        )
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=5,
            color="#222222",
            weight=1,
            fill=True,
            fill_color=sec_color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_txt, max_width=250),
        ).add_to(sector_group)
    sector_group.add_to(fmap)

    # 3. Discrete measurement points
    points_group = folium.FeatureGroup(name=f"Measurement points ({cfg['label']})", show=True)
    for p in valid_points:
        color = get_viridis_color(p["val"], cfg["min_val"], cfg["max_val"])
        popup_txt = (
            f"<b>Time:</b> {p['time']}<br>"
            f"<b>{cfg['label']}:</b> {p['val']} {cfg['unit']}<br>"
            f"<b>PCI:</b> {p['pci']} (TX {p['sector_id']})<br>"
            f"<b>Band:</b> {p['band']}"
        )
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=4,
            color="#333333",
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_txt, max_width=250),
        ).add_to(points_group)
    points_group.add_to(fmap)

    # 4. Route
    route_group = folium.FeatureGroup(name="Route", show=False)
    folium.PolyLine(route_coords, color="#1e293b", weight=2.5, opacity=0.7).add_to(route_group)
    route_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    # Legend aligned with Sionna Viridis and Sectors
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 230px;
        background-color: rgba(255, 255, 255, 0.95);
        z-index: 9999; font-size: 11px;
        border: 1px solid #cbd5e1; border-radius: 8px;
        padding: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        font-family: sans-serif;
    ">
        <b>Sionna Skala: {cfg['label']}</b><br>
        <div style="background: linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725); height: 12px; border-radius: 3px; margin: 5px 0;"></div>
        <div style="display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 8px;">
            <span>{cfg['min_val']} {cfg['unit']}</span>
            <span>{cfg['max_val']} {cfg['unit']}</span>
        </div>
        <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 6px 0;">
        <b>TX Asocijacija (PCI % 3):</b><br>
        <span style="display:inline-block; width:10px; height:10px; background-color:#800080; border-radius:50%; margin-right:4px;"></span> TX 0 (Ljubičasta)<br>
        <span style="display:inline-block; width:10px; height:10px; background-color:#00e676; border-radius:50%; margin-right:4px;"></span> TX 1 (Zelena)<br>
        <span style="display:inline-block; width:10px; height:10px; background-color:#d50000; border-radius:50%; margin-right:4px;"></span> TX 2 (Crvena)<br>
        <small style="color:#64748b;">Ukupno tačaka: {len(valid_points)}</small>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    out_name = os.path.splitext(args.csv_path)[0] + f"_{args.metric}.html"
    fmap.save(out_name)
    print(f"Uspješno generisana mapa: {out_name} (Obrađeno {len(valid_points)} tačaka).")


if __name__ == "__main__":
    main()
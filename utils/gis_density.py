"""
GIS & Spatial Kernel Density Estimation (KDE) Engine for Marine Debris Hotspots.
Generates interactive Plotly maps, density contour heatmaps, 95% error ellipse overlays,
and maritime GeoJSON/CSV exports for QGIS / ArcGIS.
"""

import json
import base64
import functools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import plotly.graph_objects as go


def compute_spatial_kde(
    lats: List[float],
    lons: List[float],
    grid_size: int = 50,
    bandwidth_deg: float = 0.0005
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 2D Gaussian Kernel Density Estimation over geographic coordinates.

    Returns:
        (lat_grid, lon_grid, density_matrix)
    """
    if len(lats) == 0:
        return np.array([]), np.array([]), np.array([])

    lat_min, lat_max = min(lats) - 0.001, max(lats) + 0.001
    lon_min, lon_max = min(lons) - 0.001, max(lons) + 0.001

    lat_lin = np.linspace(lat_min, lat_max, grid_size)
    lon_lin = np.linspace(lon_min, lon_max, grid_size)
    lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

    density = np.zeros_like(lat_grid)
    for p_lat, p_lon in zip(lats, lons):
        dist_sq = ((lat_grid - p_lat)**2 + (lon_grid - p_lon)**2) / (bandwidth_deg**2)
        density += np.exp(-0.5 * dist_sq)

    max_d = np.max(density) if np.max(density) > 0 else 1.0
    density_norm = density / max_d
    return lat_grid, lon_grid, density_norm


MAP_STYLE_PRESETS = {
    "Satellite": "satellite-streets",
    "Bathymetry": "carto-darkmatter",
    "Grayscale": "carto-positron",
}


def build_gis_hotspot_figure(
    detections: List[Dict[str, any]],
    survey_track: Optional[List[Tuple[float, float]]] = None,
    center_lat: float = 13.0827,
    center_lon: float = 80.2707,
    zoom: float = 14.5,
    map_style: str = "Satellite",
    show_track: bool = True,
    show_markers: bool = True,
    show_heatmap: bool = True,
    scan_pins: Optional[List[Dict]] = None,   # NEW: list of per-scan origin pins
) -> go.Figure:
    """
    Creates an interactive GIS Plotly Map with:
      1. Survey track line (Towfish navigation trajectory)
      2. Debris density heatmap layer (KDE Hotspots)
      3. Detected target markers with 95% error ellipse information
      4. Per-scan upload origin pins (one pin per uploaded image)

    map_style: one of MAP_STYLE_PRESETS keys ("Satellite" / "Bathymetry" / "Grayscale").
    show_track / show_markers / show_heatmap: toggle individual layer visibility.
    scan_pins: list of dicts with keys: lat, lon, label, timestamp, basin, det_count
    """
    fig = go.Figure()

    # ── 0. Per-scan Upload Origin Pins ──────────────────────────────────────
    if scan_pins:
        pin_lats = [p["lat"] for p in scan_pins]
        pin_lons = [p["lon"] for p in scan_pins]
        pin_labels = [p.get("label", f"Scan #{i+1}") for i, p in enumerate(scan_pins)]
        pin_hovers = [
            f"<b>📤 {p.get('label','Scan')}</b><br>"
            f"🕐 {p.get('timestamp','')}<br>"
            f"🌊 {p.get('basin','')}<br>"
            f"📏 {p.get('depth_m',0):.0f}m depth<br>"
            f"📍 {p['lat']:.5f}°N, {p['lon']:.5f}°E<br>"
            f"🎯 {p.get('det_count',0)} target(s)"
            for p in scan_pins
        ]
        # Latest scan gets a bright cyan pin; older ones get dimmer blue
        pin_colors = [
            "#00e5ff" if i == len(scan_pins) - 1 else "#1a6a9a"
            for i in range(len(scan_pins))
        ]
        pin_sizes = [
            16 if i == len(scan_pins) - 1 else 11
            for i in range(len(scan_pins))
        ]
        fig.add_trace(go.Scattermap(
            lat=pin_lats,
            lon=pin_lons,
            mode="markers+text",
            marker=dict(
                size=pin_sizes,
                color=pin_colors,
                opacity=0.95,
                symbol="circle",
            ),
            text=pin_labels,
            textposition="top center",
            textfont=dict(size=9, color="#a0d8ef"),
            hoverinfo="text",
            hovertext=pin_hovers,
            name="Upload Origins",
            visible=True,
        ))

    # 1. Survey Track Line
    if survey_track and len(survey_track) > 1:
        t_lats, t_lons = zip(*survey_track)
        fig.add_trace(go.Scattermap(
            lat=t_lats,
            lon=t_lons,
            mode="lines+markers",
            line=dict(width=2.5, color="#00d4ff"),
            marker=dict(size=4, color="#00d4ff"),
            name="Towfish Survey Path",
            hoverinfo="text",
            hovertext=[f"Track Point #{i+1}" for i in range(len(t_lats))],
            visible=True if show_track else "legendonly",
        ))

    # 2. Debris Density Heatmap (KDE)
    valid_dets = [d for d in detections if "latitude" in d and "longitude" in d]
    if valid_dets:
        d_lats = [d["latitude"] for d in valid_dets]
        d_lons = [d["longitude"] for d in valid_dets]
        d_confs = [d.get("conf", 0.8) for d in valid_dets]

        fig.add_trace(go.Densitymap(
            lat=d_lats,
            lon=d_lons,
            z=d_confs,
            radius=28,
            colorscale=[
                [0.0, "rgba(0, 30, 80, 0.0)"],
                [0.2, "rgba(0, 180, 216, 0.4)"],
                [0.5, "rgba(254, 217, 118, 0.7)"],
                [0.8, "rgba(253, 141, 60, 0.85)"],
                [1.0, "rgba(227, 26, 28, 0.95)"]
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Debris Density", font=dict(color="#fff", size=11)),
                tickfont=dict(color="#ddd", size=9),
                len=0.6,
                thickness=12,
                x=0.98
            ),
            name="KDE Hotspot Density",
            visible=True if show_heatmap else "legendonly",
        ))

        # 3. Individual Debris Markers
        hover_texts = []
        marker_colors = []
        for d in valid_dets:
            cname = d.get("class_name", "Target")
            conf = d.get("conf", 0.0)
            unc = d.get("uncertainty_flag", "LOW")
            err_a = d.get("error_ellipse_a", 5.0)
            err_b = d.get("error_ellipse_b", 5.0)
            channel = d.get("channel", "Port")
            range_m = d.get("ground_range_m", 0.0)

            hover_texts.append(
                f"<b>{cname}</b><br>"
                f"Confidence: {conf:.1%}<br>"
                f"Uncertainty: {unc}<br>"
                f"Range: {range_m:.1f}m ({channel})<br>"
                f"95% Error: ±{err_a:.1f}m x ±{err_b:.1f}m<br>"
                f"Lat: {d['latitude']:.5f}°N, Lon: {d['longitude']:.5f}°E"
            )
            marker_colors.append("#2ecc71" if unc == "LOW" else ("#f39c12" if unc == "MODERATE" else "#e74c3c"))

        fig.add_trace(go.Scattermap(
            lat=d_lats,
            lon=d_lons,
            mode="markers+text",
            marker=dict(
                size=12,
                color=marker_colors,
                opacity=0.9,
            ),
            text=[d.get("class_name", "") for d in valid_dets],
            textposition="top right",
            textfont=dict(size=10, color="#ffffff"),
            hoverinfo="text",
            hovertext=hover_texts,
            name="Debris Sightings",
            visible=True if show_markers else "legendonly",
        ))

    # ── Auto-fit center across ALL plotted points ────────────────────────────
    all_lats_on_map = []
    all_lons_on_map = []
    if scan_pins:
        all_lats_on_map += [p["lat"] for p in scan_pins]
        all_lons_on_map += [p["lon"] for p in scan_pins]
    if valid_dets:
        all_lats_on_map += [d["latitude"] for d in valid_dets]
        all_lons_on_map += [d["longitude"] for d in valid_dets]
    if survey_track:
        all_lats_on_map += [t[0] for t in survey_track]
        all_lons_on_map += [t[1] for t in survey_track]

    if all_lats_on_map:
        center_lat = float(np.mean(all_lats_on_map))
        center_lon = float(np.mean(all_lons_on_map))
        lat_spread = max(all_lats_on_map) - min(all_lats_on_map)
        lon_spread = max(all_lons_on_map) - min(all_lons_on_map)
        spread = max(lat_spread, lon_spread)
        if spread > 50:
            zoom = 1.5
        elif spread > 20:
            zoom = 2.5
        elif spread > 10:
            zoom = 3.5
        elif spread > 5:
            zoom = 4.5
        elif spread > 1:
            zoom = 6.0
        # else keep caller-provided zoom

    if not fig.data:
        fig.add_trace(go.Scattermap(
            lat=[center_lat],
            lon=[center_lon],
            mode="markers",
            marker=dict(size=0, opacity=0),
            hoverinfo="none",
            showlegend=False,
            name="Standby"
        ))

    fig.update_layout(
        map=dict(
            style=MAP_STYLE_PRESETS.get(map_style, "satellite-streets"),
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        paper_bgcolor="#060c18",
        plot_bgcolor="#060c18",
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(6, 12, 24, 0.82)",
            bordercolor="#1f4260",
            borderwidth=1,
            font=dict(color="#ddd", size=10)
        )
    )

    return fig



def export_detections_to_geojson(detections: List[Dict[str, any]]) -> str:

    """
    Exports debris detection list with 95% error ellipses to standard GeoJSON FeatureCollection.
    """
    features = []
    for idx, d in enumerate(detections):
        if "latitude" not in d or "longitude" not in d:
            continue

        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [d["longitude"], d["latitude"]]
            },
            "properties": {
                "id": idx + 1,
                "class_name": d.get("class_name", "Target"),
                "confidence": d.get("conf", 0.0),
                "uncertainty_flag": d.get("uncertainty_flag", "LOW"),
                "uncertainty_variance": d.get("uncertainty_variance", 0.0),
                "error_ellipse_a_m": d.get("error_ellipse_a", 0.0),
                "error_ellipse_b_m": d.get("error_ellipse_b", 0.0),
                "ground_range_m": d.get("ground_range_m", 0.0),
                "channel": d.get("channel", "Port"),
                "source": d.get("source", "Akhet-AI")
            }
        }
        features.append(feat)

    collection = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }
    return json.dumps(collection, indent=2)


def export_detections_to_csv(detections: List[Dict[str, any]]) -> str:
    """
    Exports debris detection list to CSV format for marine survey reports.
    """
    headers = [
        "id", "class_name", "confidence", "uncertainty_flag", "latitude", "longitude",
        "ground_range_m", "channel", "error_ellipse_a_m", "error_ellipse_b_m", "source"
    ]
    lines = [",".join(headers)]
    for idx, d in enumerate(detections):
        if "latitude" not in d:
            continue
        row = [
            str(idx + 1),
            f'"{d.get("class_name", "Target")}"',
            f'{d.get("conf", 0.0):.3f}',
            d.get("uncertainty_flag", "LOW"),
            f'{d.get("latitude", 0.0):.6f}',
            f'{d.get("longitude", 0.0):.6f}',
            f'{d.get("ground_range_m", 0.0):.2f}',
            d.get("channel", "Port"),
            f'{d.get("error_ellipse_a", 0.0):.2f}',
            f'{d.get("error_ellipse_b", 0.0):.2f}',
            f'"{d.get("source", "Akhet-AI")}"'
        ]
        lines.append(",".join(row))

    return "\n".join(lines)



def build_leaflet_map_html(
    detections: List[Dict[str, Any]],
    survey_track: Optional[List[Tuple[float, float]]] = None,
    center_lat: float = 12.834,
    center_lon: float = 80.242,
    zoom: int = 14,
    resurvey_zones: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Generates an interactive standalone Leaflet HTML map matching Slide 3 Tech Stack.
    Renders CartoDB DarkMatter basemap, AUV tracklines, 3-tier color-coded markers,
    95% covariance error ellipses, and SCQI Resurvey Recommended zones.
    """
    if detections and "latitude" in detections[0]:
        center_lat = detections[0]["latitude"]
        center_lon = detections[0]["longitude"]

    track_coords_js = json.dumps([[p[0], p[1]] for p in survey_track]) if survey_track else "[]"

    markers_js = []
    for i, d in enumerate(detections):
        lat = d.get("latitude", center_lat)
        lon = d.get("longitude", center_lon)
        cls_name = d.get("class", "Marine Debris")
        tier = d.get("reliability_tier", "PROBABLE").upper()
        conf = d.get("confidence", 0.85)
        conf_pct = f"{conf*100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"
        err_a = d.get("error_ellipse_a_m", 1.5)
        err_b = d.get("error_ellipse_b_m", 0.9)
        length_m = d.get("length_m", 2.0)
        width_m = d.get("width_m", 0.8)

        if tier == "CONFIRMED":
            color = "#00e676"
        elif tier == "PROBABLE":
            color = "#ffc107"
        else:
            color = "#ff5252"

        popup_html = f"""
        <div style='font-family: sans-serif; color: #fff; min-width: 160px;'>
            <div style='font-size: 11px; color: #00bcd4; font-weight: bold;'>#{i+1:02d} &bull; {tier}</div>
            <div style='font-size: 14px; font-weight: bold; margin: 2px 0; color: #ffffff;'>{cls_name}</div>
            <div style='font-size: 12px; color: #7b9bb3;'>Confidence: <b style='color:#00e5ff;'>{conf_pct}</b></div>
            <div style='font-size: 12px; color: #7b9bb3;'>Dimensions: {length_m:.1f}m x {width_m:.1f}m</div>
            <div style='font-size: 11px; color: #4a7590; margin-top: 4px;'>WGS-84: {lat:.5f}, {lon:.5f}</div>
            <div style='font-size: 11px; color: #ffc107;'>95% Error Ellipse: &plusmn;{err_a:.1f}m x &plusmn;{err_b:.1f}m</div>
        </div>
        """
        escaped_popup = popup_html.replace("\n", " ").replace("'", "\'")

        marker_entry = f"""
        (function() {{
            var circle = L.circleMarker([{lat}, {lon}], {{
                radius: 8,
                fillColor: '{color}',
                color: '#ffffff',
                weight: 1.5,
                opacity: 1.0,
                fillOpacity: 0.9
            }}).addTo(map);
            circle.bindPopup('{escaped_popup}');

            // Draw 95% position error ellipse as transparent bounding circle
            L.circle([{lat}, {lon}], {{
                radius: {err_a},
                color: '{color}',
                weight: 1,
                dashArray: '3, 4',
                fillColor: '{color}',
                fillOpacity: 0.15
            }}).addTo(map);
        }})();
        """
        markers_js.append(marker_entry)

    all_markers_code = "\n".join(markers_js)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body, #leaflet-map {{ height: 100%; width: 100%; margin: 0; padding: 0; background: #060b13; }}
        .leaflet-popup-content-wrapper {{ background: rgba(8, 20, 34, 0.95); border: 1px solid rgba(0, 188, 212, 0.4); border-radius: 8px; box-shadow: 0 4px 18px rgba(0,0,0,0.8); }}
        .leaflet-popup-tip {{ background: rgba(8, 20, 34, 0.95); }}
    </style>
</head>
<body>
    <div id="leaflet-map"></div>
    <script>
        var map = L.map('leaflet-map', {{
            center: [{center_lat}, {center_lon}],
            zoom: {zoom},
            maxBounds: [[-85.0, -180.0], [85.0, 180.0]],
            maxBoundsViscosity: 1.0,
            attributionControl: false
        }});

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 19,
            subdomains: 'abcd',
            noWrap: true,
            bounds: [[-85.0, -180.0], [85.0, 180.0]]
        }}).addTo(map);

        var trackCoords = {track_coords_js};
        if (trackCoords.length > 1) {{
            L.polyline(trackCoords, {{
                color: '#00d4ff',
                weight: 2.5,
                opacity: 0.85,
                dashArray: '4, 4'
            }}).addTo(map);
        }}

        {all_markers_code}
    </script>
</body>
</html>"""
    return html


@functools.lru_cache(maxsize=8)
def _get_texture_b64(filename: str) -> str:
    path = Path(__file__).resolve().parent.parent / filename
    if not path.exists():
        path = Path(__file__).resolve().parent / filename
    if not path.exists():
        path = Path.cwd() / filename
    if path.exists():
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


def build_3d_globe_html(
    detections: List[Dict[str, Any]],
    survey_track: Optional[List[Tuple[float, float]]] = None,
    center_lat: float = 13.0827,
    center_lon: float = 80.2707,
    height_px: int = 620,
) -> str:
    """
    Builds a high-performance, photorealistic 3D Earth Globe using Three.js and WebGL.
    Features:
      - 2K NASA Blue Marble Earth with bump topology and specular ocean reflection
      - Atmospheric Rayleigh blue scattering rim glow
      - 2,000+ deep space starfield particles
      - Neon 3D towfish survey trackline with waypoint nodes
      - 3D Seabed debris pins color-coded by calibrated reliability:
          Confirmed (Green), Probable (Yellow), Uncertain (Red)
      - Pulsing sonar radar rings expanding outward across the ocean surface
      - Concentric KDE thermal density halo over survey cluster
      - Interactive glassmorphic HUD:
          * Focus AUV Survey (smooth camera fly-to lerp directly to coordinates)
          * Orbital auto-rotation toggle
          * Reset to Global view
          * Day / Night city lights toggle
          * Interactive hover tooltip HUD card with acoustic metrics
    """
    clean_dets = []
    for d in detections:
        if "latitude" in d and "longitude" in d:
            clean_dets.append({
                "latitude": float(d["latitude"]),
                "longitude": float(d["longitude"]),
                "class_name": str(d.get("class_name", "Target")),
                "conf": float(d.get("conf", 0.8)),
                "channel": str(d.get("channel", "Port")),
                "ground_range_m": float(d.get("ground_range_m", 0.0)),
                "error_ellipse_a": float(d.get("error_ellipse_a", 5.0)),
                "length_m": float(d.get("length_m", 2.0)),
                "width_m": float(d.get("width_m", 1.0)),
                "uncertainty_flag": str(d.get("uncertainty_flag", "LOW")),
            })

    clean_track = []
    if survey_track:
        for pt in survey_track:
            clean_track.append([float(pt[0]), float(pt[1])])

    if clean_dets:
        center_lat = float(np.mean([d["latitude"] for d in clean_dets]))
        center_lon = float(np.mean([d["longitude"] for d in clean_dets]))

    dets_json = json.dumps(clean_dets)
    track_json = json.dumps(clean_track)

    day_b64 = _get_texture_b64("earth_blue_marble.jpg")
    topo_b64 = _get_texture_b64("earth_topology.png")
    water_b64 = _get_texture_b64("earth_water.png")
    night_b64 = _get_texture_b64("earth_night.jpg")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Akhet 3D Marine Digital Globe</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: radial-gradient(circle at center, #071426 0%, #02060f 100%);
            overflow: hidden;
            width: 100vw;
            height: 100vh;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #e0f2fe;
            user-select: none;
        }}
        #canvas-container {{
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }}
        .hud-glass {{
            position: absolute;
            background: rgba(6, 18, 33, 0.85);
            border: 1px solid rgba(0, 229, 255, 0.3);
            border-radius: 10px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), inset 0 0 12px rgba(0, 229, 255, 0.08);
            z-index: 10;
        }}
        .control-panel {{
            top: 14px;
            left: 14px;
            padding: 12px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-width: 320px;
        }}
        .control-title {{
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #00e5ff;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .control-meta {{
            font-size: 11px;
            color: #7fa1b8;
            line-height: 1.4;
        }}
        .btn-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 4px;
        }}
        .hud-btn {{
            background: rgba(0, 229, 255, 0.12);
            border: 1px solid rgba(0, 229, 255, 0.45);
            color: #e0f2fe;
            padding: 6px 11px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }}
        .hud-btn:hover {{
            background: rgba(0, 229, 255, 0.32);
            border-color: #00e5ff;
            box-shadow: 0 0 12px rgba(0, 229, 255, 0.5);
            transform: translateY(-1px);
        }}
        .hud-btn.active {{
            background: #00bcd4;
            color: #020813;
            border-color: #00e5ff;
        }}
        .legend-panel {{
            bottom: 14px;
            left: 14px;
            padding: 8px 14px;
            display: flex;
            align-items: center;
            gap: 14px;
            font-size: 11px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }}
        #hover-tooltip {{
            position: absolute;
            display: none;
            background: rgba(5, 15, 28, 0.94);
            border: 1px solid #00e5ff;
            border-radius: 8px;
            padding: 9px 13px;
            font-size: 11px;
            color: #fff;
            pointer-events: none;
            z-index: 100;
            box-shadow: 0 8px 24px rgba(0,0,0,0.8), 0 0 14px rgba(0, 229, 255, 0.3);
            max-width: 250px;
        }}
        .tooltip-tier {{
            font-size: 10px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
            margin-bottom: 4px;
        }}
        .tooltip-name {{
            font-size: 13px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 4px;
        }}
        .tooltip-row {{
            display: flex;
            justify-content: space-between;
            gap: 10px;
            color: #8da8ba;
            margin-top: 2px;
        }}
        .tooltip-row b {{
            color: #e0f2fe;
        }}
    </style>
</head>
<body>
    <div id="canvas-container"></div>

    <div class="hud-glass control-panel">
        <div class="control-title">
            <span style="font-size: 15px;">🌐</span> Akhet 3D Marine Globe
        </div>
        <div class="control-meta">
            Survey Zone: <b>{center_lat:.4f}°N, {center_lon:.4f}°E</b><br>
            Acoustic Targets: <b>{len(clean_dets)} Sightings{" (Standby)" if not clean_dets else ""}</b> | EPSG:4326
        </div>
        <div class="btn-row">
            <button class="hud-btn" id="btn-focus">🎯 Focus AUV Survey</button>
            <button class="hud-btn" id="btn-rotate">🔄 Orbit</button>
            <button class="hud-btn" id="btn-reset">🌍 Global</button>
            <button class="hud-btn" id="btn-daynight">🌓 Night</button>
        </div>
    </div>

    <div class="hud-glass legend-panel" style="display: {'flex' if clean_dets else 'none'};">
        <div class="legend-item"><span class="legend-dot" style="background:#00e676;box-shadow:0 0 8px #00e676;"></span> Confirmed (≥75%)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#ffc107;box-shadow:0 0 8px #ffc107;"></span> Probable (45-74%)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#ff5252;box-shadow:0 0 8px #ff5252;"></span> Uncertain (&lt;45%)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#00e5ff;box-shadow:0 0 8px #00e5ff;"></span> AUV Track</div>
    </div>

    <div id="hover-tooltip"></div>

    <script>
        const container = document.getElementById('canvas-container');
        const tooltip = document.getElementById('hover-tooltip');

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 3000);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true, powerPreference: 'high-performance' }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.35;
        renderer.outputEncoding = THREE.sRGBEncoding;
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.minDistance = 100.2;
        controls.maxDistance = 550;
        controls.enablePan = false; // ROTATING AXIS LOCKED AT CENTER OF EARTH (0,0,0) - NO DRIFT
        controls.target.set(0, 0, 0); // ROTATING AXIS EXACTLY AT (0,0,0)
        controls.autoRotate = false;
        controls.autoRotateSpeed = 0.8;

        // Multi-Source Bright Illumination (Vibrant Earth & Ocean)
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
        scene.add(ambientLight);

        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x1d3d63, 0.85);
        scene.add(hemiLight);

        const sunLight = new THREE.DirectionalLight(0xffffff, 1.45);
        sunLight.position.set(250, 180, 200);
        scene.add(sunLight);

        const camLight = new THREE.DirectionalLight(0xffffff, 0.75);
        camera.add(camLight);
        scene.add(camera);

        const oceanFillLight = new THREE.DirectionalLight(0x00d4ff, 0.45);
        oceanFillLight.position.set(-250, -100, -150);
        scene.add(oceanFillLight);

        // Coordinate conversion helper (WGS-84 to 3D Cartesian)
        function latLonToVec3(lat, lon, radius) {{
            const phi = (90 - lat) * (Math.PI / 180);
            const theta = (lon + 180) * (Math.PI / 180);
            const x = -(radius * Math.sin(phi) * Math.cos(theta));
            const z = radius * Math.sin(phi) * Math.sin(theta);
            const y = radius * Math.cos(phi);
            return new THREE.Vector3(x, y, z);
        }}

        // Texture Loader
        const texLoader = new THREE.TextureLoader();
        const dayB64 = "{day_b64}";
        const topoB64 = "{topo_b64}";
        const waterB64 = "{water_b64}";
        const nightB64 = "{night_b64}";

        const dayTex = dayB64 ? texLoader.load("data:image/jpeg;base64," + dayB64) : texLoader.load("https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg");
        const topoTex = topoB64 ? texLoader.load("data:image/png;base64," + topoB64) : null;
        const waterTex = waterB64 ? texLoader.load("data:image/png;base64," + waterB64) : null;
        const nightTex = nightB64 ? texLoader.load("data:image/jpeg;base64," + nightB64) : null;

        if (dayTex) dayTex.encoding = THREE.sRGBEncoding;
        if (nightTex) nightTex.encoding = THREE.sRGBEncoding;

        // Earth Mesh (Luminous, High-Contrast Texture with specular ocean)
        const earthRadius = 100;
        const earthGeo = new THREE.SphereGeometry(earthRadius, 72, 72);
        const earthMat = new THREE.MeshStandardMaterial({{
            color: 0xffffff, // Pure white base so texture is 100% bright and vivid
            map: dayTex,
            bumpMap: topoTex,
            bumpScale: 0.04,
            roughnessMap: waterTex,
            roughness: 0.35,
            metalness: 0.04,
            emissive: new THREE.Color(0x0a1d33),
            emissiveIntensity: 0.18
        }});
        const earthMesh = new THREE.Mesh(earthGeo, earthMat);
        scene.add(earthMesh);

        // Visual Central Rotational Axis (straight through center of earth 0,0,0)
        const axisGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, -125, 0),
            new THREE.Vector3(0, 125, 0)
        ]);
        const axisMat = new THREE.LineDashedMaterial({{
            color: 0x00e5ff,
            dashSize: 3,
            gapSize: 2,
            transparent: true,
            opacity: 0.45
        }});
        const axisLine = new THREE.Line(axisGeo, axisMat);
        axisLine.computeLineDistances();
        scene.add(axisLine);

        // Polar Rings indicating rotational axis poles
        [100.5, -100.5].forEach(yPos => {{
            const pRingGeo = new THREE.RingGeometry(1.5, 2.8, 24);
            const pRingMat = new THREE.MeshBasicMaterial({{ color: 0x00e5ff, side: THREE.DoubleSide, transparent: true, opacity: 0.65 }});
            const pRingMesh = new THREE.Mesh(pRingGeo, pRingMat);
            pRingMesh.position.set(0, yPos, 0);
            pRingMesh.rotation.x = Math.PI / 2;
            scene.add(pRingMesh);
        }});

        // Atmospheric Rayleigh Glow
        const atmosGeo = new THREE.SphereGeometry(earthRadius * 1.025, 64, 64);
        const atmosMat = new THREE.ShaderMaterial({{
            vertexShader: `
                varying vec3 vNormal;
                void main() {{
                    vNormal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }}
            `,
            fragmentShader: `
                varying vec3 vNormal;
                void main() {{
                    float intensity = pow(0.64 - dot(vNormal, vec3(0, 0, 1.0)), 2.2);
                    gl_FragColor = vec4(0.0, 0.78, 1.0, 1.0) * intensity;
                }}
            `,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide,
            transparent: true
        }});
        const atmosMesh = new THREE.Mesh(atmosGeo, atmosMat);
        scene.add(atmosMesh);

        // Deep Space Starfield
        const starGeo = new THREE.BufferGeometry();
        const starCount = 2200;
        const starPos = new Float32Array(starCount * 3);
        for (let i = 0; i < starCount; i++) {{
            const r = 550 + Math.random() * 800;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos((Math.random() * 2) - 1);
            starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            starPos[i * 3 + 1] = r * Math.cos(phi);
            starPos[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
        }}
        starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
        const starMat = new THREE.PointsMaterial({{ color: 0x90caf9, size: 1.2, transparent: true, opacity: 0.75 }});
        scene.add(new THREE.Points(starGeo, starMat));

        // Towfish Survey Trajectory
        const trackData = {track_json};
        if (trackData.length > 1) {{
            const trackPts = trackData.map(pt => latLonToVec3(pt[0], pt[1], earthRadius + 0.6));
            const trackCurve = new THREE.CatmullRomCurve3(trackPts);
            const trackGeo = new THREE.TubeGeometry(trackCurve, trackPts.length * 6, 0.22, 8, false);
            const trackMat = new THREE.MeshBasicMaterial({{ color: 0x00f0ff }});
            scene.add(new THREE.Mesh(trackGeo, trackMat));

            trackPts.forEach(p => {{
                const wpGeo = new THREE.SphereGeometry(0.45, 12, 12);
                const wpMat = new THREE.MeshBasicMaterial({{ color: 0x00e5ff }});
                const wpMesh = new THREE.Mesh(wpGeo, wpMat);
                wpMesh.position.copy(p);
                scene.add(wpMesh);
            }});
        }}

        // Debris Sightings & Pulsing Sonar Rings
        const dets = {dets_json};
        const markerMeshes = [];
        const radarRings = [];

        dets.forEach((d, idx) => {{
            const lat = d.latitude;
            const lon = d.longitude;
            const conf = d.conf || 0.8;
            const unc = d.uncertainty_flag || 'LOW';

            let colorHex = 0x00e676;
            let tier = 'CONFIRMED';
            if (conf < 0.45 || unc === 'HIGH') {{
                colorHex = 0xff5252;
                tier = 'UNCERTAIN';
            }} else if (conf < 0.75 || unc === 'MODERATE') {{
                colorHex = 0xffc107;
                tier = 'PROBABLE';
            }}

            const pSurface = latLonToVec3(lat, lon, earthRadius + 0.15);
            const pTip = latLonToVec3(lat, lon, earthRadius + 4.2);

            // Vertical 3D Pin Stem
            const stemGeo = new THREE.BufferGeometry().setFromPoints([pSurface, pTip]);
            const stemMat = new THREE.LineBasicMaterial({{ color: colorHex, linewidth: 2 }});
            scene.add(new THREE.Line(stemGeo, stemMat));

            // Pin Beacon Head
            const headGeo = new THREE.SphereGeometry(0.85, 16, 16);
            const headMat = new THREE.MeshStandardMaterial({{
                color: colorHex,
                emissive: colorHex,
                emissiveIntensity: 0.65,
                metalness: 0.2,
                roughness: 0.3
            }});
            const headMesh = new THREE.Mesh(headGeo, headMat);
            headMesh.position.copy(pTip);
            headMesh.userData = {{
                det: d,
                tier: tier,
                colorHex: colorHex,
                name: d.class_name || ('Target #' + (idx + 1))
            }};
            scene.add(headMesh);
            markerMeshes.push(headMesh);

            // Pulsing Sonar Radar Ring
            const ringGeo = new THREE.RingGeometry(0.3, 0.7, 24);
            const ringMat = new THREE.MeshBasicMaterial({{
                color: colorHex,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.7
            }});
            const ringMesh = new THREE.Mesh(ringGeo, ringMat);
            ringMesh.position.copy(pSurface);
            const normal = pSurface.clone().normalize();
            ringMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
            scene.add(ringMesh);

            radarRings.push({{
                mesh: ringMesh,
                scale: 1.0 + (idx % 6) * 0.4,
                maxScale: 3.8,
                speed: 0.035 + (idx % 3) * 0.012
            }});
        }});

        // Thermal KDE Cluster Disc (Rendered only when detections are present)
        if (dets.length > 0) {{
            const centerSurf = latLonToVec3({center_lat}, {center_lon}, earthRadius + 0.2);
            const heatGeo = new THREE.RingGeometry(0.5, 4.2, 32);
            const heatMat = new THREE.MeshBasicMaterial({{
                color: 0xff3d00,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.38
            }});
            const heatMesh = new THREE.Mesh(heatGeo, heatMat);
            heatMesh.position.copy(centerSurf);
            const heatNormal = centerSurf.clone().normalize();
            heatMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), heatNormal);
            scene.add(heatMesh);
        }}

        // Initial Camera Position (Looking towards Survey Zone with center rigidly at 0,0,0)
        const initCamPos = latLonToVec3({center_lat}, {center_lon}, 260);
        camera.position.copy(initCamPos);
        controls.target.set(0, 0, 0); // ROTATIONAL AXIS AT EXACT CENTER OF EARTH (0,0,0)

        // Camera Smooth Fly-To Controller (Rotational Axis Remains Locked at 0,0,0)
        let isFlying = false;
        let targetCamPos = null;
        let targetLookAt = new THREE.Vector3(0, 0, 0); // NEVER LEAVE CENTER (0,0,0)

        function flyTo(targetLat, targetLon, distance) {{
            // Move camera radially toward target location, keeping target strictly at center (0,0,0)
            const targetDir = latLonToVec3(targetLat, targetLon, 1.0).normalize();
            targetCamPos = targetDir.clone().multiplyScalar(distance);
            targetLookAt.set(0, 0, 0); // ROTATING AXIS IS FIXED AT CENTER OF EARTH
            isFlying = true;
        }}

        // Button Listeners
        document.getElementById('btn-focus').addEventListener('click', () => {{
            flyTo({center_lat}, {center_lon}, 108);
        }});

        const btnRotate = document.getElementById('btn-rotate');
        btnRotate.addEventListener('click', () => {{
            controls.autoRotate = !controls.autoRotate;
            btnRotate.classList.toggle('active', controls.autoRotate);
        }});

        document.getElementById('btn-reset').addEventListener('click', () => {{
            const dir = latLonToVec3({center_lat}, {center_lon}, 1.0).normalize();
            targetCamPos = dir.clone().multiplyScalar(260);
            targetLookAt.set(0, 0, 0);
            isFlying = true;
        }});

        let isNight = false;
        const btnDayNight = document.getElementById('btn-daynight');
        btnDayNight.addEventListener('click', () => {{
            isNight = !isNight;
            if (isNight && nightTex) {{
                earthMat.map = nightTex;
                earthMat.emissive = new THREE.Color(0xffffff);
                earthMat.emissiveMap = nightTex;
                earthMat.emissiveIntensity = 0.85;
                atmosMesh.visible = false;
                btnDayNight.innerHTML = '☀️ Day';
            }} else {{
                earthMat.map = dayTex;
                earthMat.emissive = new THREE.Color(0x000000);
                earthMat.emissiveMap = null;
                earthMat.emissiveIntensity = 0.0;
                atmosMesh.visible = true;
                btnDayNight.innerHTML = '🌓 Night';
            }}
            earthMat.needsUpdate = true;
        }});

        // Raycaster for Hover Tooltips
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        window.addEventListener('mousemove', (e) => {{
            mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(markerMeshes);

            if (intersects.length > 0) {{
                const hit = intersects[0].object;
                const d = hit.userData.det;
                const tier = hit.userData.tier;
                document.body.style.cursor = 'pointer';

                let tierColor = '#00e676';
                if (tier === 'PROBABLE') tierColor = '#ffc107';
                if (tier === 'UNCERTAIN') tierColor = '#ff5252';

                tooltip.innerHTML = `
                    <div class="tooltip-tier" style="background:${{tierColor}}25; color:${{tierColor}}; border:1px solid ${{tierColor}};">
                        ${{tier}}
                    </div>
                    <div class="tooltip-name">${{hit.userData.name}}</div>
                    <div class="tooltip-row"><span>Confidence:</span><b>${{((d.conf || 0.8) * 100).toFixed(1)}}%</b></div>
                    <div class="tooltip-row"><span>WGS-84:</span><b>${{(d.latitude || 0).toFixed(4)}}°N, ${{(d.longitude || 0).toFixed(4)}}°E</b></div>
                    <div class="tooltip-row"><span>Range:</span><b>${{(d.ground_range_m || 0).toFixed(1)}}m (${{d.channel || 'Port'}})</b></div>
                    <div class="tooltip-row"><span>95% Err:</span><b>±${{(d.error_ellipse_a || 5).toFixed(1)}}m</b></div>
                `;
                tooltip.style.display = 'block';
                tooltip.style.left = Math.min(window.innerWidth - 260, e.clientX + 14) + 'px';
                tooltip.style.top = Math.min(window.innerHeight - 180, e.clientY + 14) + 'px';
            }} else {{
                document.body.style.cursor = 'default';
                tooltip.style.display = 'none';
            }}
        }});

        // Responsive Resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // Render Loop
        function animate() {{
            requestAnimationFrame(animate);

            // Animate pulsing sonar rings
            radarRings.forEach(r => {{
                r.scale += r.speed;
                if (r.scale > r.maxScale) r.scale = 1.0;
                r.mesh.scale.set(r.scale, r.scale, 1.0);
                r.mesh.material.opacity = Math.max(0.0, 0.75 * (1.0 - (r.scale - 1.0) / (r.maxScale - 1.0)));
            }});

            // Camera Fly-To Lerp (Axis remains locked at center 0,0,0)
            if (isFlying && targetCamPos) {{
                camera.position.lerp(targetCamPos, 0.05);
                controls.target.set(0, 0, 0);
                if (camera.position.distanceTo(targetCamPos) < 1.0) {{
                    isFlying = false;
                    controls.target.set(0, 0, 0);
                }}
            }}

            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>"""
    return html


def build_maplibre_globe_html(
    detections: List[Dict[str, Any]],
    survey_track: Optional[List[Tuple[float, float]]] = None,
    center_lat: float = 13.0827,
    center_lon: float = 80.2707,
    height_px: int = 640,
) -> str:
    """
    Generates a high-resolution 2D/3D Satellite Map using MapLibre GL v4 with dynamic
    streaming satellite tiles (Esri World Imagery) and street/boundary overlays.
    Enforces a SINGLE map canvas with FIXED boundaries (renderWorldCopies: false, strict maxBounds)
    to completely prevent infinite horizontal repetitions or panning into the void.
    Enables fluid zooming all the way from space down to sub-meter street/harbor level (zoom 19+).
    """
    clean_features = []
    for idx, d in enumerate(detections):
        if "latitude" in d and "longitude" in d:
            conf = float(d.get("conf", 0.8))
            unc = str(d.get("uncertainty_flag", "LOW"))
            color = "#00e676"
            tier = "CONFIRMED"
            if conf < 0.45 or unc == "HIGH":
                color = "#ff5252"
                tier = "UNCERTAIN"
            elif conf < 0.75 or unc == "MODERATE":
                color = "#ffc107"
                tier = "PROBABLE"

            clean_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(d["longitude"]), float(d["latitude"])]
                },
                "properties": {
                    "id": idx + 1,
                    "class_name": str(d.get("class_name", "Target")),
                    "conf": conf,
                    "tier": tier,
                    "color": color,
                    "channel": str(d.get("channel", "Port")),
                    "ground_range_m": float(d.get("ground_range_m", 0.0)),
                    "error_ellipse_a": float(d.get("error_ellipse_a", 5.0)),
                }
            })

    track_coords = []
    if survey_track:
        for pt in survey_track:
            track_coords.append([float(pt[1]), float(pt[0])])  # [lon, lat] for GeoJSON

    # Calculate precise geographic bounding box for the survey
    all_lats = [d["latitude"] for d in detections if "latitude" in d]
    all_lons = [d["longitude"] for d in detections if "longitude" in d]
    if survey_track:
        all_lats.extend([pt[0] for pt in survey_track])
        all_lons.extend([pt[1] for pt in survey_track])

    if all_lats and all_lons:
        min_lat = min(all_lats)
        max_lat = max(all_lats)
        min_lon = min(all_lons)
        max_lon = max(all_lons)
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
    else:
        min_lat, max_lat = center_lat - 0.05, center_lat + 0.05
        min_lon, max_lon = center_lon - 0.05, center_lon + 0.05

    geojson_data = {
        "type": "FeatureCollection",
        "features": clean_features
    }
    geojson_str = json.dumps(geojson_data)
    track_json_str = json.dumps(track_coords)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Akhet Single Satellite Map (Fixed Bounds)</title>
    <link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" />
    <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body, html {{
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #030712;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #e0f2fe;
        }}
        #map {{
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            background: #030712;
        }}
        .map-frame-border {{
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            border: 1px solid rgba(0, 229, 255, 0.25);
            box-shadow: inset 0 0 28px rgba(3, 7, 18, 0.85);
            z-index: 5;
        }}
        .hud-glass {{
            position: absolute;
            background: rgba(6, 18, 33, 0.90);
            border: 1px solid rgba(0, 229, 255, 0.35);
            border-radius: 10px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.75), inset 0 0 14px rgba(0, 229, 255, 0.1);
            z-index: 10;
        }}
        .control-panel {{
            top: 14px;
            left: 14px;
            padding: 12px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-width: 340px;
        }}
        .control-title {{
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #00e5ff;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .control-meta {{
            font-size: 11px;
            color: #8da8ba;
            line-height: 1.45;
        }}
        .btn-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 4px;
        }}
        .hud-btn {{
            background: rgba(0, 229, 255, 0.12);
            border: 1px solid rgba(0, 229, 255, 0.45);
            color: #e0f2fe;
            padding: 6px 11px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }}
        .hud-btn:hover {{
            background: rgba(0, 229, 255, 0.32);
            border-color: #00e5ff;
            box-shadow: 0 0 12px rgba(0, 229, 255, 0.5);
            transform: translateY(-1px);
        }}
        .legend-panel {{
            bottom: 14px;
            left: 14px;
            padding: 8px 14px;
            display: flex;
            align-items: center;
            gap: 14px;
            font-size: 11px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }}
        .maplibregl-popup-content {{
            background: rgba(5, 16, 30, 0.95) !important;
            border: 1px solid #00e5ff !important;
            border-radius: 8px !important;
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.8), 0 0 16px rgba(0, 229, 255, 0.3) !important;
            padding: 10px 14px !important;
            color: #fff !important;
        }}
        .maplibregl-popup-anchor-bottom .maplibregl-popup-tip {{
            border-top-color: rgba(5, 16, 30, 0.95) !important;
        }}
        .maplibregl-popup-close-button {{
            color: #8da8ba !important;
            font-size: 16px !important;
            padding: 4px 8px !important;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="map-frame-border"></div>

    <div class="hud-glass control-panel">
        <div class="control-title">
            <span style="font-size: 15px;">🛰️</span> Single Bounded Map
        </div>
        <div class="control-meta">
            Single World &bull; <b>No Infinite Repeat</b> &bull; Fixed Limits<br>
            Resolution: <b>Sub-Meter Satellite (Zoom 0 &rarr; 19.5)</b><br>
            Mission Center: <b>{center_lat:.4f}&deg;N, {center_lon:.4f}&deg;E</b>
        </div>
        <div class="btn-row">
            <button class="hud-btn" id="btn-focus">🎯 Street Focus</button>
            <button class="hud-btn" id="btn-lock">🔒 Lock Survey Bounds</button>
            <button class="hud-btn" id="btn-fit-world">🌍 Single World</button>
            <button class="hud-btn" id="btn-tilt">📐 3D Tilt</button>
        </div>
    </div>

    <div class="hud-glass legend-panel">
        <div class="legend-item"><span class="legend-dot" style="background:#00e676;box-shadow:0 0 8px #00e676;"></span> Confirmed (&ge;75%)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#ffc107;box-shadow:0 0 8px #ffc107;"></span> Probable (45-74%)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#ff5252;box-shadow:0 0 8px #ff5252;"></span> Uncertain (&lt;45%)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#00e5ff;box-shadow:0 0 8px #00e5ff;"></span> Towfish Track</div>
    </div>

    <script>
        const map = new maplibregl.Map({{
            container: 'map',
            renderWorldCopies: false, // CRITICAL: NEVER REPEAT WORLD HORIZONTALLY - SINGLE MAP ONLY!
            maxBounds: [
                [-180.0, -85.0],
                [180.0, 85.0]
            ], // STRICT FIXED BOUNDARIES: NO INFINITE SCROLLING INTO VOID
            maxBoundsViscosity: 1.0, // Hard stop at boundaries
            style: {{
                version: 8,
                sources: {{
                    'satellite-tiles': {{
                        type: 'raster',
                        tiles: [
                            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}'
                        ],
                        tileSize: 256,
                        maxzoom: 19,
                        attribution: '&copy; Esri, Maxar, Earthstar Geographics'
                    }},
                    'street-overlay': {{
                        type: 'raster',
                        tiles: [
                            'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}'
                        ],
                        tileSize: 256,
                        maxzoom: 19
                    }}
                }},
                layers: [
                    {{
                        id: 'background',
                        type: 'background',
                        paint: {{
                            'background-color': '#030712'
                        }}
                    }},
                    {{
                        id: 'satellite-layer',
                        type: 'raster',
                        source: 'satellite-tiles',
                        minzoom: 0,
                        maxzoom: 20
                    }},
                    {{
                        id: 'street-labels-layer',
                        type: 'raster',
                        source: 'street-overlay',
                        minzoom: 7,
                        maxzoom: 20
                    }}
                ]
            }},
            center: [{center_lon}, {center_lat}],
            zoom: 13.5, // Start directly in the survey corridor with high resolution!
            minZoom: 1.5,
            maxZoom: 19.5, // ZOOMS ALL THE WAY DOWN TO SUB-METER STREET LEVEL
            pitch: 0,
            bearing: 0
        }});

        map.addControl(new maplibregl.NavigationControl({{ visualizePitch: true }}), 'top-right');

        map.on('load', () => {{
            // Automatically fit survey area bounds so user sees full corridor immediately
            map.fitBounds([
                [{min_lon - 0.025}, {min_lat - 0.025}],
                [{max_lon + 0.025}, {max_lat + 0.025}]
            ], {{ padding: 50, maxZoom: 16.0, duration: 1000 }});

            // 1. Towfish Trackline
            const trackCoords = {track_json_str};
            if (trackCoords && trackCoords.length > 1) {{
                map.addSource('towfish-track', {{
                    type: 'geojson',
                    data: {{
                        type: 'Feature',
                        geometry: {{
                            type: 'LineString',
                            coordinates: trackCoords
                        }}
                    }}
                }});
                map.addLayer({{
                    id: 'track-glow',
                    type: 'line',
                    source: 'towfish-track',
                    paint: {{
                        'line-color': '#00e5ff',
                        'line-width': 7,
                        'line-opacity': 0.45,
                        'line-blur': 3
                    }}
                }});
                map.addLayer({{
                    id: 'track-core',
                    type: 'line',
                    source: 'towfish-track',
                    paint: {{
                        'line-color': '#00f0ff',
                        'line-width': 3,
                        'line-dasharray': [3, 2]
                    }}
                }});
            }}

            // 2. Debris Detections GeoJSON
            const detectionsGeoJSON = {geojson_str};
            map.addSource('debris-src', {{
                type: 'geojson',
                data: detectionsGeoJSON
            }});

            // 95% Position Covariance Error Rings
            map.addLayer({{
                id: 'debris-error-rings',
                type: 'circle',
                source: 'debris-src',
                paint: {{
                    'circle-radius': ['interpolate', ['linear'], ['zoom'], 2, 4, 10, 14, 15, 28, 19, 50],
                    'circle-color': ['get', 'color'],
                    'circle-opacity': 0.22,
                    'circle-stroke-width': 1.5,
                    'circle-stroke-color': ['get', 'color'],
                    'circle-stroke-dasharray': [2, 2]
                }}
            }});

            // Core Debris Markers
            map.addLayer({{
                id: 'debris-markers',
                type: 'circle',
                source: 'debris-src',
                paint: {{
                    'circle-radius': ['interpolate', ['linear'], ['zoom'], 2, 4, 10, 7, 15, 11, 19, 16],
                    'circle-color': ['get', 'color'],
                    'circle-opacity': 0.95,
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#ffffff'
                }}
            }});

            // Popups on Click
            map.on('click', 'debris-markers', (e) => {{
                const f = e.features[0];
                const p = f.properties;
                const coords = f.geometry.coordinates.slice();

                new maplibregl.Popup({{ className: 'akhet-popup' }})
                    .setLngLat(coords)
                    .setHTML(`
                        <div style="font-family: sans-serif; color: #fff; min-width: 170px;">
                            <div style="font-size: 10px; font-weight: 700; color: ${{p.color}}; letter-spacing: 0.05em;">${{p.tier}}</div>
                            <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 3px 0;">${{p.class_name}}</div>
                            <div style="font-size: 11px; color: #8da8ba; margin-top: 3px;">Confidence: <b style="color:#00e5ff;">${{(p.conf * 100).toFixed(1)}}%</b></div>
                            <div style="font-size: 11px; color: #8da8ba;">Range: ${{p.ground_range_m}}m (${{p.channel}})</div>
                            <div style="font-size: 11px; color: #8da8ba;">95% Err: &plusmn;${{p.error_ellipse_a}}m</div>
                            <div style="font-size: 10px; color: #7fa1b8; margin-top: 4px;">WGS-84: ${{coords[1].toFixed(5)}}&deg;N, ${{coords[0].toFixed(5)}}&deg;E</div>
                        </div>
                    `)
                    .addTo(map);
            }});

            map.on('mouseenter', 'debris-markers', () => {{ map.getCanvas().style.cursor = 'pointer'; }});
            map.on('mouseleave', 'debris-markers', () => {{ map.getCanvas().style.cursor = ''; }});
        }});

        // 1. Street Focus Button
        document.getElementById('btn-focus').addEventListener('click', () => {{
            map.flyTo({{
                center: [{center_lon}, {center_lat}],
                zoom: 17.5, // SUB-METER STREET / HARBOR LEVEL
                pitch: 55,
                bearing: -15,
                speed: 1.2,
                curve: 1.4
            }});
        }});

        // 2. Lock / Unlock Survey Bounds Toggle
        let isLockedToSurvey = false;
        const btnLock = document.getElementById('btn-lock');
        btnLock.addEventListener('click', () => {{
            isLockedToSurvey = !isLockedToSurvey;
            if (isLockedToSurvey) {{
                // Clamps bounds rigidly to the survey mission corridor
                map.setMaxBounds([
                    [{min_lon - 0.08}, {min_lat - 0.08}],
                    [{max_lon + 0.08}, {max_lat + 0.08}]
                ]);
                map.fitBounds([
                    [{min_lon - 0.02}, {min_lat - 0.02}],
                    [{max_lon + 0.02}, {max_lat + 0.02}]
                ], {{ padding: 30, duration: 1000 }});
                btnLock.innerText = '🔓 Unlock World';
                btnLock.style.background = 'rgba(0, 230, 118, 0.25)';
                btnLock.style.borderColor = '#00e676';
            }} else {{
                // Single world fixed bounds
                map.setMaxBounds([
                    [-180.0, -85.0],
                    [180.0, 85.0]
                ]);
                btnLock.innerText = '🔒 Lock Survey Bounds';
                btnLock.style.background = 'rgba(0, 229, 255, 0.12)';
                btnLock.style.borderColor = 'rgba(0, 229, 255, 0.45)';
            }}
        }});

        // 3. Single World Button (Fits single map without repeating)
        document.getElementById('btn-fit-world').addEventListener('click', () => {{
            map.setMaxBounds([
                [-180.0, -85.0],
                [180.0, 85.0]
            ]);
            map.fitBounds([
                [-175.0, -75.0],
                [175.0, 75.0]
            ], {{ padding: 25, duration: 1200 }});
        }});

        // 4. 3D Tilt View Toggle
        let isTilted = false;
        document.getElementById('btn-tilt').addEventListener('click', () => {{
            isTilted = !isTilted;
            map.easeTo({{
                pitch: isTilted ? 60 : 0,
                bearing: isTilted ? -25 : 0,
                duration: 900
            }});
        }});
    </script>
</body>
</html>"""
    return html


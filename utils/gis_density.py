"""
GIS & Spatial Kernel Density Estimation (KDE) Engine for Marine Debris Hotspots.
Generates interactive Plotly maps, density contour heatmaps, 95% error ellipse overlays,
and maritime GeoJSON/CSV exports for QGIS / ArcGIS.
"""

import json
from typing import Dict, List, Optional, Tuple, Union
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


def build_gis_hotspot_figure(
    detections: List[Dict[str, any]],
    survey_track: Optional[List[Tuple[float, float]]] = None,
    center_lat: float = 13.0827,
    center_lon: float = 80.2707,
    zoom: float = 14.5
) -> go.Figure:
    """
    Creates an interactive GIS Plotly Map with:
      1. Survey track line (Towfish navigation trajectory)
      2. Debris density heatmap layer (KDE Hotspots)
      3. Detected target markers with 95% error ellipse information
    """
    fig = go.Figure()

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
            hovertext=[f"Track Point #{i+1}" for i in range(len(t_lats))]
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
            name="KDE Hotspot Density"
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
            name="Debris Sightings"
        ))

        center_lat = float(np.mean(d_lats))
        center_lon = float(np.mean(d_lons))

    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
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
            bgcolor="rgba(6, 12, 24, 0.8)",
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

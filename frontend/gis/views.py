"""GIS view: WGS-84 debris points, detection tracks, KDE hotspots, SCQI heatmap, resurvey areas, exports."""

from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

from backend.gis.export import to_csv, to_geojson, to_json


def build_map(result: Dict[str, Any], show: Dict[str, bool]) -> go.Figure:
    fig = go.Figure()
    dets: List[Dict[str, Any]] = [d for d in result["detections"] if d.get("latitude") is not None]

    if show.get("scqi"):
        cells = result["gis"]["scqi_heatmap"]["cells"]
        if cells:
            fig.add_trace(go.Scattermap(
                lat=[c["latitude"] for c in cells], lon=[c["longitude"] for c in cells], mode="markers", name="SCQI",
                marker=dict(size=22, opacity=0.55, color=[c["scqi"] for c in cells], colorscale="RdYlGn", cmin=0, cmax=100,
                            showscale=True, colorbar=dict(title="SCQI", len=0.5)),
                text=[f"SCQI {c['scqi']:.0f} ({c['n_frames']} frame)" for c in cells], hoverinfo="text"))
    if show.get("resurvey"):
        for a in result["gis"]["resurvey"]["areas"]:
            x1, y1, x2, y2 = a["bbox"]
            pad = 0.0002
            fig.add_trace(go.Scattermap(lon=[x1 - pad, x2 + pad, x2 + pad, x1 - pad, x1 - pad],
                                        lat=[y1 - pad, y1 - pad, y2 + pad, y2 + pad, y1 - pad], mode="lines",
                                        line=dict(color="#ff5252", width=3), name="Resurvey area",
                                        text=f"Resurvey: SCQI {a['mean_scqi']:.0f}; {', '.join(a['reasons'])}", hoverinfo="text"))
    if show.get("kde"):
        hs = result["gis"]["kde"]["hotspots"]
        if hs:
            fig.add_trace(go.Scattermap(lat=[h["latitude"] for h in hs], lon=[h["longitude"] for h in hs], mode="markers",
                                        name="KDE hotspot", marker=dict(size=[20 + 25 * h["peak_density"] for h in hs],
                                                                        color="rgba(255,60,60,0.35)"),
                                        text=[f"Hotspot {h['hotspot_id']}: {h['n_objects']} object(s)" for h in hs],
                                        hoverinfo="text"))
    if show.get("tracks"):
        by_track: Dict[Any, List[Dict[str, Any]]] = {}
        for d in dets:
            by_track.setdefault(d["track_id"], []).append(d)
        for tid, pts in by_track.items():
            if len(pts) > 1:
                fig.add_trace(go.Scattermap(lat=[p["latitude"] for p in pts], lon=[p["longitude"] for p in pts],
                                            mode="lines", line=dict(width=2, color="#00e676"), name=f"Track {tid}"))
    for kind, color in (("known", "#ffc107"), ("unknown", "#e040fb")):
        pts = [d for d in dets if d["detection_type"] == kind]
        if pts and show.get(kind):
            fig.add_trace(go.Scattermap(
                lat=[p["latitude"] for p in pts], lon=[p["longitude"] for p in pts], mode="markers",
                name=f"{kind.title()} debris" if kind == "known" else "Unknown anomaly", marker=dict(size=13, color=color),
                text=[f"<b>{p['class_name']}</b><br>conf {p['confidence']:.2f}<br>track {p['track_id']}<br>"
                      f"+/-{p['positional_uncertainty_m']:.1f} m<br>{p['latitude']:.5f}, {p['longitude']:.5f}" for p in pts],
                hoverinfo="text"))
    lat = [d["latitude"] for d in dets] or [c["latitude"] for c in result["gis"]["scqi_heatmap"]["cells"]] or [13.08]
    lon = [d["longitude"] for d in dets] or [c["longitude"] for c in result["gis"]["scqi_heatmap"]["cells"]] or [80.27]
    fig.update_layout(map=dict(style="carto-darkmatter", center=dict(lat=sum(lat) / len(lat), lon=sum(lon) / len(lon)), zoom=16.5),
                      margin=dict(l=0, r=0, t=0, b=0), height=560, legend=dict(bgcolor="rgba(10,20,35,0.8)", font=dict(color="#ddd")))
    return fig


def render_gis(result: Dict[str, Any]) -> None:
    c = st.columns(4)
    show = {"known": c[0].checkbox("Known debris", True), "unknown": c[1].checkbox("Unknown anomalies", True),
            "tracks": c[2].checkbox("Detection tracks", True), "kde": c[3].checkbox("KDE hotspots", True)}
    c = st.columns(2)
    show["scqi"] = c[0].checkbox("SCQI heatmap", True)
    show["resurvey"] = c[1].checkbox("Resurvey areas", True)
    st.plotly_chart(build_map(result, show), use_container_width=True)
    g = result["gis"]
    m = st.columns(3)
    m[0].metric("Mapped objects", len(result["objects"]))
    m[1].metric("KDE hotspots", len(g["kde"]["hotspots"]))
    m[2].metric("Resurvey areas", len(g["resurvey"]["areas"]))
    st.caption("CRS: WGS-84 / EPSG:4326. Positions carry a 95% positional-uncertainty radius (see the detection table).")
    d = st.columns(3)
    recs = result["detections"]
    d[0].download_button("Export CSV", to_csv(recs), f"marineguard_{result['run_id']}.csv", "text/csv", use_container_width=True)
    d[1].download_button("Export JSON", to_json(recs), f"marineguard_{result['run_id']}.json", "application/json",
                         use_container_width=True)
    d[2].download_button("Export GeoJSON", to_geojson(recs), f"marineguard_{result['run_id']}.geojson", "application/geo+json",
                         use_container_width=True)

"""GIS analytics: KDE debris hotspots, SCQI heatmap and resurvey areas (all WGS-84 / EPSG:4326)."""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from utils.gis_density import compute_spatial_kde


def _cell_polygon(lat: float, lon: float, dlat: float, dlon: float) -> List[List[float]]:
    return [[lon - dlon, lat - dlat], [lon + dlon, lat - dlat], [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
            [lon - dlon, lat - dlat]]


def kde_hotspots(points: List[Dict[str, Any]], cfg) -> Dict[str, Any]:
    """points: dicts with latitude/longitude. Returns density grid summary + hotspot clusters."""
    pts = [p for p in points if p.get("latitude") is not None and p.get("longitude") is not None]
    if not pts:
        return {"n_points": 0, "hotspots": [], "geojson": {"type": "FeatureCollection", "features": []}}
    lats, lons = [p["latitude"] for p in pts], [p["longitude"] for p in pts]
    g = cfg.gis
    lat_g, lon_g, dens = compute_spatial_kde(lats, lons, grid_size=int(g.kde_grid), bandwidth_deg=float(g.kde_bandwidth_deg))
    thr = float(np.percentile(dens, g.hotspot_percentile))
    thr = max(thr, 0.5 * float(dens.max()) if len(pts) < 5 else thr)          # few points: keep only the peaks
    mask = (dens >= thr).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    dlat = float(abs(lat_g[1, 0] - lat_g[0, 0])) / 2
    dlon = float(abs(lon_g[0, 1] - lon_g[0, 0])) / 2
    hotspots, features = [], []
    for k in range(1, n):
        ys, xs = np.where(labels == k)
        w = dens[ys, xs]
        clat, clon = float(np.average(lat_g[ys, xs], weights=w)), float(np.average(lon_g[ys, xs], weights=w))
        inside = sum(1 for la, lo in zip(lats, lons)
                     if lat_g[ys, xs].min() - dlat <= la <= lat_g[ys, xs].max() + dlat
                     and lon_g[ys, xs].min() - dlon <= lo <= lon_g[ys, xs].max() + dlon)
        hs = {"hotspot_id": k, "latitude": clat, "longitude": clon, "peak_density": float(w.max()), "n_objects": inside,
              "cells": int(len(ys))}
        hotspots.append(hs)
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [clon, clat]},
                         "properties": {**hs, "kind": "kde_hotspot"}})
    hotspots.sort(key=lambda h: -h["peak_density"])
    return {"n_points": len(pts), "threshold_density": thr, "hotspots": hotspots,
            "geojson": {"type": "FeatureCollection", "features": features},
            "density": {"lat_min": float(lat_g.min()), "lat_max": float(lat_g.max()), "lon_min": float(lon_g.min()),
                        "lon_max": float(lon_g.max()), "grid": int(g.kde_grid)}}


def scqi_heatmap(frames: List[Dict[str, Any]], cfg) -> Dict[str, Any]:
    """frames: [{'latitude','longitude','scqi'}] -> gridded mean SCQI cells + polygon GeoJSON."""
    cell = float(cfg.gis.scqi_cell_deg)
    acc: Dict[tuple, List[float]] = {}
    for f in frames:
        if f.get("latitude") is None:
            continue
        key = (round(f["latitude"] / cell), round(f["longitude"] / cell))
        acc.setdefault(key, []).append(float(f["scqi"]))
    cells, feats = [], []
    for (i, j), v in sorted(acc.items()):
        lat, lon = i * cell, j * cell
        c = {"latitude": lat, "longitude": lon, "scqi": float(np.mean(v)), "n_frames": len(v)}
        cells.append(c)
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_cell_polygon(lat, lon, cell / 2, cell / 2)]},
                      "properties": {**c, "kind": "scqi_cell"}})
    return {"cells": cells, "geojson": {"type": "FeatureCollection", "features": feats}}


def resurvey_areas(frames: List[Dict[str, Any]], cfg) -> Dict[str, Any]:
    """Consecutive low-SCQI frames merge into one resurvey area (bounding box + reasons)."""
    thr = float(cfg.scqi.resurvey_threshold)
    areas, cur = [], []
    for f in frames + [None]:
        if f is not None and f.get("latitude") is not None and f["scqi"] < thr:
            cur.append(f)
            continue
        if cur:
            lats, lons = [c["latitude"] for c in cur], [c["longitude"] for c in cur]
            reasons = sorted({r for c in cur for r in c.get("reasons", [])})
            areas.append({"frame_ids": [c["frame_id"] for c in cur], "mean_scqi": float(np.mean([c["scqi"] for c in cur])),
                          "bbox": [min(lons), min(lats), max(lons), max(lats)], "reasons": reasons})
            cur = []
    feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_cell_polygon(
        (a["bbox"][1] + a["bbox"][3]) / 2, (a["bbox"][0] + a["bbox"][2]) / 2,
        max((a["bbox"][3] - a["bbox"][1]) / 2, float(cfg.gis.scqi_cell_deg) / 2),
        max((a["bbox"][2] - a["bbox"][0]) / 2, float(cfg.gis.scqi_cell_deg) / 2))]},
        "properties": {**{k: v for k, v in a.items() if k != "bbox"}, "kind": "resurvey_area"}} for a in areas]
    return {"threshold": thr, "areas": areas, "geojson": {"type": "FeatureCollection", "features": feats}}

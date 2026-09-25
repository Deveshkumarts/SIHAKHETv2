"""CSV / JSON / GeoJSON export of detection records (every field, incl. track id + model version)."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List

CSV_COLUMNS = ["survey_id", "run_id", "frame_id", "timestamp", "class_name", "detection_type", "confidence",
               "x1", "y1", "x2", "y2", "mask_area_px", "anomaly_score", "reconstruction_error", "track_id",
               "latitude", "longitude", "positional_uncertainty_m", "snr_db", "scqi", "model_version", "config_hash"]


def _flat(r: Dict[str, Any]) -> Dict[str, Any]:
    bbox = r.get("bbox") or [None] * 4
    mv = r.get("model_version")
    return {**{k: r.get(k) for k in CSV_COLUMNS if k not in ("x1", "y1", "x2", "y2", "model_version")},
            "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
            "model_version": json.dumps(mv, sort_keys=True) if isinstance(mv, dict) else mv}


def to_csv(records: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in records:
        w.writerow(_flat(r))
    return buf.getvalue()


def to_json(records: List[Dict[str, Any]], indent: int = 2) -> str:
    return json.dumps({"count": len(records), "detections": records}, indent=indent, default=str)


def to_geojson(records: List[Dict[str, Any]]) -> str:
    feats = []
    for r in records:
        if r.get("latitude") is None or r.get("longitude") is None:
            continue
        props = {k: v for k, v in r.items() if k not in ("latitude", "longitude")}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
                      "properties": props})
    return json.dumps({"type": "FeatureCollection",
                       "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                       "features": feats}, indent=2, default=str)

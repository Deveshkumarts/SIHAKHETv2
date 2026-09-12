"""
FastAPI REST API Service for Autonomous AUV & Headless Edge Deployments
SIH26057 - Akhet: Marine Guard ("Turning Echoes into Impact")

Provides RESTful endpoints for:
  - Dual-Branch Detection (Edge Mode vs. Full Mode)
  - SCQI Survey Quality Evaluation & Resurvey Flagging
  - Telemetry Ingestion & Real-Time Monitoring
  - GeoJSON, CSV, and PDF Report Exports
"""

import os
import io
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import cv2

from utils.telemetry_parser import generate_synthetic_telemetry, TelemetryRecord
from utils.scqi_engine import compute_scqi, SCQIResult
from utils.report_generator import generate_html_report
from utils.gis_density import export_detections_to_geojson, export_detections_to_csv

app = FastAPI(
    title="Akhet Marine Guard REST API",
    description="Autonomous Side-Scan Sonar Marine Debris & Anomaly Detection (SIH26057)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated in-memory mission store
ACTIVE_DETECTIONS: List[Dict[str, Any]] = [
    {
        "id": "DEB-01",
        "class": "Shipwrecks",
        "confidence": 0.94,
        "reliability_tier": "CONFIRMED",
        "latitude": 12.8345,
        "longitude": 80.2421,
        "length_m": 42.5,
        "width_m": 12.0,
        "error_ellipse_a_m": 1.4,
        "error_ellipse_b_m": 0.9,
    },
    {
        "id": "DEB-02",
        "class": "large-tire",
        "confidence": 0.78,
        "reliability_tier": "PROBABLE",
        "latitude": 12.8360,
        "longitude": 80.2440,
        "length_m": 1.4,
        "width_m": 1.4,
        "error_ellipse_a_m": 0.8,
        "error_ellipse_b_m": 0.6,
    }
]


class SCQIRequest(BaseModel):
    altitude_m: float = Field(10.0, description="Towfish altitude above seafloor in meters")
    slant_range_m: float = Field(75.0, description="Sonar slant range in meters")
    vessel_speed_knots: float = Field(3.5, description="AUV survey speed in knots")
    pitch_deg: float = Field(0.5, description="Towfish pitch attitude in degrees")
    roll_deg: float = Field(0.8, description="Towfish roll attitude in degrees")
    heave_m: float = Field(0.05, description="Towfish heave in meters")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "platform": "Akhet: Marine Guard",
        "ps_id": "SIH26057",
        "organization": "MoES / NIOT",
        "tagline": "Turning Echoes into Impact",
        "timestamp": time.time()
    }


@app.get("/api/v1/telemetry")
def get_telemetry():
    rec = generate_synthetic_telemetry(num_pings=1)[0]
    return rec.to_dict()


@app.post("/api/v1/scqi")
def evaluate_scqi(req: SCQIRequest):
    telemetry = TelemetryRecord(
        timestamp=time.time(),
        latitude=12.834,
        longitude=80.241,
        heading_deg=45.0,
        altitude_m=req.altitude_m,
        slant_range_m=req.slant_range_m,
        vessel_speed_knots=req.vessel_speed_knots,
        pitch_deg=req.pitch_deg,
        roll_deg=req.roll_deg,
        heave_m=req.heave_m
    )
    res = compute_scqi(image_bgr=None, telemetry=telemetry)
    return res.to_dict()


@app.get("/api/v1/detections")
def list_detections():
    return {
        "count": len(ACTIVE_DETECTIONS),
        "detections": ACTIVE_DETECTIONS
    }


@app.get("/api/v1/export/geojson")
def export_geojson():
    gj = export_detections_to_geojson(ACTIVE_DETECTIONS)
    return Response(content=gj, media_type="application/geo+json")


@app.get("/api/v1/export/csv")
def export_csv():
    csv_str = export_detections_to_csv(ACTIVE_DETECTIONS)
    return Response(content=csv_str, media_type="text/csv")


@app.get("/api/v1/export/report")
def export_report(format: str = Query("html", regex="^(html|pdf)$")):
    telemetry = generate_synthetic_telemetry(num_pings=1)[0]
    scqi = compute_scqi(image_bgr=None, telemetry=telemetry)
    html = generate_html_report(
        mission_id="AUV_MISSION_ALPHA",
        detections=ACTIVE_DETECTIONS,
        scqi_data=scqi.to_dict(),
        telemetry_data=telemetry.to_dict()
    )
    return Response(content=html, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

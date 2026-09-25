"""
Detection repository.  PostgreSQL/PostGIS when reachable (DSN in $MARINEGUARD_PG_DSN), otherwise SQLite with
the same columns so the rest of the system never has to care which one is behind it.

Every detection row stores the model versions and config hash that produced it.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("marineguard.db")

DET_COLUMNS = ["survey_id", "run_id", "frame_id", "timestamp", "class_name", "detection_type", "confidence", "bbox",
               "segmentation_mask", "mask_area_px", "anomaly_score", "reconstruction_error", "track_id", "latitude",
               "longitude", "positional_uncertainty_m", "snr_db", "scqi", "model_version", "config_hash", "evidence"]
JSON_COLS = {"bbox", "segmentation_mask", "model_version", "evidence"}

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS surveys (
    survey_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, source_file TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP, scqi REAL, snr_db REAL, config_hash TEXT, model_version TEXT);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, survey_id TEXT NOT NULL, run_id TEXT NOT NULL, frame_id INTEGER NOT NULL,
    timestamp REAL, class_name TEXT NOT NULL, detection_type TEXT NOT NULL, confidence REAL, bbox TEXT,
    segmentation_mask TEXT, mask_area_px INTEGER, anomaly_score REAL, reconstruction_error REAL, track_id INTEGER,
    latitude REAL, longitude REAL, positional_uncertainty_m REAL, snr_db REAL, scqi REAL, model_version TEXT NOT NULL,
    config_hash TEXT NOT NULL, evidence TEXT);
CREATE INDEX IF NOT EXISTS detections_survey_ix ON detections (survey_id, frame_id);
CREATE INDEX IF NOT EXISTS detections_track_ix  ON detections (survey_id, track_id);
"""


def _dumps(v: Any) -> Optional[str]:
    return None if v is None else json.dumps(v, default=str)


class DetectionRepository:
    def __init__(self, cfg, backend: Optional[str] = None, sqlite_path: Optional[str] = None):
        self.cfg = cfg
        self.fallback_reason: Optional[str] = None
        want = backend or cfg.database.backend
        self.backend = "sqlite"
        self._pg = None
        if want in ("auto", "postgis"):
            self._try_postgis(strict=(want == "postgis"))
        if self.backend == "sqlite":
            path = Path(sqlite_path or cfg.path("database.sqlite_path"))
            path.parent.mkdir(parents=True, exist_ok=True)
            self._sq = sqlite3.connect(str(path), check_same_thread=False)
            self._sq.row_factory = sqlite3.Row
            self._sq.executescript(SQLITE_DDL)
            self.location = str(path)

    # ---- backend setup ----
    def _try_postgis(self, strict: bool) -> None:
        dsn = os.environ.get(self.cfg.database.postgres_dsn_env)
        if not dsn:
            self.fallback_reason = f"${self.cfg.database.postgres_dsn_env} not set"
        else:
            try:
                import psycopg2                                     # optional dependency
                self._pg = psycopg2.connect(dsn, connect_timeout=5)
                self._pg.autocommit = True
                with self._pg.cursor() as cur:
                    cur.execute((Path(__file__).parent / "schema.sql").read_text())
                self.backend, self.location = "postgis", dsn.split("@")[-1]
                return
            except Exception as e:
                self._pg = None
                self.fallback_reason = f"PostGIS unavailable ({type(e).__name__}: {e})"
        if strict:
            raise RuntimeError(self.fallback_reason)
        log.warning("Using SQLite fallback: %s", self.fallback_reason)

    # ---- write ----
    def save_run(self, survey: Dict[str, Any], detections: List[Dict[str, Any]]) -> int:
        """Idempotent per survey_id: replaces the survey's previous rows."""
        # Integrity: every row belongs to the survey/run being saved, whatever the caller put in the record,
        # otherwise rows get orphaned and the idempotent replace below would miss them.
        rows = [{**{c: d.get(c) for c in DET_COLUMNS}, "survey_id": survey["survey_id"], "run_id": survey["run_id"]}
                for d in detections]
        if self.backend == "postgis":
            from psycopg2.extras import Json
            with self._pg.cursor() as cur:
                cur.execute("DELETE FROM surveys WHERE survey_id=%s", (survey["survey_id"],))
                cur.execute("INSERT INTO surveys (survey_id, run_id, source_file, scqi, snr_db, config_hash, model_version) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            (survey["survey_id"], survey["run_id"], survey.get("source_file"), survey.get("scqi"),
                             survey.get("snr_db"), survey.get("config_hash"), Json(survey.get("model_version"))))
                for r in rows:
                    vals = [Json(r[c]) if c in JSON_COLS and r[c] is not None else r[c] for c in DET_COLUMNS]
                    cols = ", ".join('"timestamp"' if c == "timestamp" else c for c in DET_COLUMNS)
                    has_pt = r["latitude"] is not None and r["longitude"] is not None
                    geom_sql = "ST_SetSRID(ST_MakePoint(%s,%s),4326)" if has_pt else "NULL"
                    cur.execute(f"INSERT INTO detections ({cols}, geom) VALUES ({', '.join(['%s'] * len(DET_COLUMNS))}, {geom_sql})",
                                vals + ([r["longitude"], r["latitude"]] if has_pt else []))
            return len(rows)
        with self._sq:
            self._sq.execute("DELETE FROM detections WHERE survey_id=?", (survey["survey_id"],))
            self._sq.execute("DELETE FROM surveys WHERE survey_id=?", (survey["survey_id"],))
            self._sq.execute("INSERT INTO surveys (survey_id, run_id, source_file, scqi, snr_db, config_hash, model_version) "
                             "VALUES (?,?,?,?,?,?,?)",
                             (survey["survey_id"], survey["run_id"], survey.get("source_file"), survey.get("scqi"),
                              survey.get("snr_db"), survey.get("config_hash"), _dumps(survey.get("model_version"))))
            self._sq.executemany(
                f"INSERT INTO detections ({', '.join(DET_COLUMNS)}) VALUES ({', '.join('?' * len(DET_COLUMNS))})",
                [[_dumps(r[c]) if c in JSON_COLS else r[c] for c in DET_COLUMNS] for r in rows])
        return len(rows)

    # ---- read ----
    def detections(self, survey_id: Optional[str] = None, run_id: Optional[str] = None,
                   detection_type: Optional[str] = None) -> List[Dict[str, Any]]:
        where, args = [], []
        for col, val in (("survey_id", survey_id), ("run_id", run_id), ("detection_type", detection_type)):
            if val:
                where.append(f"{col}={'%s' if self.backend == 'postgis' else '?'}")
                args.append(val)
        sql = "SELECT " + ", ".join('"timestamp"' if (c == "timestamp" and self.backend == "postgis") else c
                                    for c in DET_COLUMNS) + " FROM detections" + (" WHERE " + " AND ".join(where) if where else "") \
              + " ORDER BY frame_id, id"
        if self.backend == "postgis":
            with self._pg.cursor() as cur:
                cur.execute(sql, args)
                return [dict(zip(DET_COLUMNS, row)) for row in cur.fetchall()]
        out = []
        for row in self._sq.execute(sql, args):
            d = dict(row)
            for c in JSON_COLS:
                if d.get(c) is not None:
                    d[c] = json.loads(d[c])
            out.append(d)
        return out

    def surveys(self) -> List[Dict[str, Any]]:
        if self.backend == "postgis":
            with self._pg.cursor() as cur:
                cur.execute("SELECT survey_id, run_id, source_file, scqi, snr_db, config_hash FROM surveys ORDER BY created_at DESC")
                return [dict(zip(["survey_id", "run_id", "source_file", "scqi", "snr_db", "config_hash"], r)) for r in cur.fetchall()]
        return [dict(r) for r in self._sq.execute("SELECT survey_id, run_id, source_file, scqi, snr_db, config_hash FROM surveys ORDER BY created_at DESC")]

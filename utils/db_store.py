"""
Spatial Persistence Layer for Marine Survey Telemetry & Debris Detections.
SQLite / PostGIS-compatible database abstraction for indexing, spatial querying,
and mission record management.
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union


class SurveyDatabase:
    """
    Local Spatial Database for Side-Scan Sonar Surveys & Detections.
    Stores WGS-84 coordinates, uncertainty metrics, error ellipses, and review status.
    """

    def __init__(self, db_path: Union[str, Path] = "outputs/survey_database.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Missions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS missions (
                mission_id TEXT PRIMARY KEY,
                mission_name TEXT,
                vessel_name TEXT,
                start_time REAL,
                total_pings INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0
            );
            """)

            # Detections table with spatial indexing
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_id TEXT,
                class_name TEXT,
                confidence REAL,
                uncertainty_flag TEXT,
                uncertainty_variance REAL,
                latitude REAL,
                longitude REAL,
                ground_range_m REAL,
                channel TEXT,
                error_ellipse_a REAL,
                error_ellipse_b REAL,
                error_ellipse_phi REAL,
                reviewed_status TEXT DEFAULT 'UNREVIEWED',
                timestamp REAL,
                FOREIGN KEY (mission_id) REFERENCES missions(mission_id)
            );
            """)

            # Spatial Indices for fast geographic bounding box queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_spatial ON detections(latitude, longitude);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mission ON detections(mission_id);")
            conn.commit()

    def record_mission(self, mission_id: str, mission_name: str = "Survey Line Alpha", vessel_name: str = "RV Akhet"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO missions (mission_id, mission_name, vessel_name, start_time)
            VALUES (?, ?, ?, ?);
            """, (mission_id, mission_name, vessel_name, time.time()))
            conn.commit()

    def save_detections(self, detections: List[Dict[str, Any]], mission_id: str = "mission_default"):
        self.record_mission(mission_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for d in detections:
                cursor.execute("""
                INSERT INTO detections (
                    mission_id, class_name, confidence, uncertainty_flag,
                    uncertainty_variance, latitude, longitude, ground_range_m,
                    channel, error_ellipse_a, error_ellipse_b, error_ellipse_phi,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    mission_id,
                    d.get("class_name", "Target"),
                    float(d.get("conf", 0.0)),
                    d.get("uncertainty_flag", "LOW"),
                    float(d.get("uncertainty_variance", 0.0)),
                    float(d.get("latitude", 0.0)),
                    float(d.get("longitude", 0.0)),
                    float(d.get("ground_range_m", 0.0)),
                    d.get("channel", "Port"),
                    float(d.get("error_ellipse_a", 0.0)),
                    float(d.get("error_ellipse_b", 0.0)),
                    float(d.get("error_ellipse_phi", 0.0)),
                    time.time()
                ))

            # Update mission detection count
            cursor.execute("""
            UPDATE missions
            SET total_detections = (SELECT COUNT(*) FROM detections WHERE mission_id = ?)
            WHERE mission_id = ?;
            """, (mission_id, mission_id))
            conn.commit()

    def get_all_detections(self, mission_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if mission_id:
                cursor.execute("SELECT * FROM detections WHERE mission_id = ? ORDER BY id DESC;", (mission_id,))
            else:
                cursor.execute("SELECT * FROM detections ORDER BY id DESC;")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def query_spatial_bounds(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM detections
            WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?;
            """, (min_lat, max_lat, min_lon, max_lon))
            return [dict(r) for r in cursor.fetchall()]

    def update_review_status(self, detection_id: int, new_status: str, corrected_class: Optional[str] = None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if corrected_class:
                cursor.execute("""
                UPDATE detections
                SET reviewed_status = ?, class_name = ?
                WHERE id = ?;
                """, (new_status, corrected_class, detection_id))
            else:
                cursor.execute("""
                UPDATE detections
                SET reviewed_status = ?
                WHERE id = ?;
                """, (new_status, detection_id))
            conn.commit()

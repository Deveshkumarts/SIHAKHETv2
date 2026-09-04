"""
PostgreSQL + PostGIS Production Spatial Database Adapter & SQL Generator.
Implements:
  1. PostGIS Schema DDL Generation with ST_Point GEOMETRY(Point, 4326)
  2. GIST Spatial Indexing for Fast Geographic Queries
  3. Seamless PostgreSQL Connection with Graceful SQLite Spatial Store Fallback
  4. PostGIS SQL Dump Exporter for Maritime GIS Servers (GeoServer / QGIS Server)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from utils.db_store import SurveyDatabase


POSTGIS_SCHEMA_SQL = """-- PostGIS Spatial Schema for Akhet Marine AI Platform
-- Enable PostGIS spatial extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Missions Table
CREATE TABLE IF NOT EXISTS sonar_missions (
    mission_id VARCHAR(64) PRIMARY KEY,
    mission_name VARCHAR(128),
    vessel_name VARCHAR(128),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    total_pings INT DEFAULT 0,
    total_detections INT DEFAULT 0
);

-- 2. Marine Debris Detections Table with Native PostGIS WGS-84 Geometry
CREATE TABLE IF NOT EXISTS marine_debris_detections (
    id SERIAL PRIMARY KEY,
    mission_id VARCHAR(64) REFERENCES sonar_missions(mission_id),
    class_name VARCHAR(64) NOT NULL,
    confidence REAL NOT NULL,
    fused_confidence REAL,
    uncertainty_flag VARCHAR(32) DEFAULT 'LOW',
    uncertainty_variance REAL DEFAULT 0.0,
    ground_range_m REAL,
    channel VARCHAR(16),
    error_ellipse_a REAL,
    error_ellipse_b REAL,
    error_ellipse_phi REAL,
    geom GEOMETRY(Point, 4326) NOT NULL,
    reviewed_status VARCHAR(32) DEFAULT 'UNREVIEWED',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. High-Performance GIST Spatial Index for PostGIS Bounding Box & k-NN Proximity
CREATE INDEX IF NOT EXISTS idx_debris_geom_gist ON marine_debris_detections USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_debris_mission ON marine_debris_detections(mission_id);
"""


class PostGISAdapter:
    """
    Production PostgreSQL/PostGIS database adapter.
    Attempts PostgreSQL connection via psycopg2/sqlalchemy if credentials provided;
    otherwise falls back gracefully to the embedded SQLite spatial store.
    """

    def __init__(
        self,
        pg_host: Optional[str] = None,
        pg_port: int = 5432,
        pg_database: str = "akhet_marine",
        pg_user: str = "postgres",
        pg_password: Optional[str] = None,
        sqlite_fallback_path: str = "outputs/survey_database.db"
    ):
        self.host = pg_host or os.environ.get("POSTGRES_HOST")
        self.port = pg_port or int(os.environ.get("POSTGRES_PORT", 5432))
        self.database = pg_database or os.environ.get("POSTGRES_DB", "akhet_marine")
        self.user = pg_user or os.environ.get("POSTGRES_USER", "postgres")
        self.password = pg_password or os.environ.get("POSTGRES_PASSWORD", "")

        self.sqlite_db = SurveyDatabase(sqlite_fallback_path)
        self.is_pg_connected = False
        self._test_pg_connection()

    def _test_pg_connection(self):
        if self.host:
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=self.host, port=self.port, dbname=self.database,
                    user=self.user, password=self.password, connect_timeout=3
                )
                conn.close()
                self.is_pg_connected = True
            except Exception:
                self.is_pg_connected = False
        else:
            self.is_pg_connected = False

    def save_detections(self, detections: List[Dict[str, Any]], mission_id: str = "survey_alpha"):
        """
        Saves detections to PostGIS if connected; otherwise uses SQLite spatial database.
        """
        if self.is_pg_connected:
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=self.host, port=self.port, dbname=self.database,
                    user=self.user, password=self.password
                )
                cur = conn.cursor()
                # Insert mission
                cur.execute("""
                INSERT INTO sonar_missions (mission_id, mission_name)
                VALUES (%s, %s) ON CONFLICT (mission_id) DO NOTHING;
                """, (mission_id, f"Survey {mission_id}"))

                # Insert detections with PostGIS ST_SetSRID(ST_MakePoint(lon, lat), 4326)
                for d in detections:
                    lat = float(d.get("latitude", 0.0))
                    lon = float(d.get("longitude", 0.0))
                    cur.execute("""
                    INSERT INTO marine_debris_detections (
                        mission_id, class_name, confidence, fused_confidence,
                        uncertainty_flag, uncertainty_variance, ground_range_m,
                        channel, error_ellipse_a, error_ellipse_b, error_ellipse_phi,
                        geom
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                    );
                    """, (
                        mission_id, d.get("class_name", "Target"),
                        float(d.get("conf", 0.0)), float(d.get("fused_confidence", d.get("conf", 0.0))),
                        d.get("uncertainty_flag", "LOW"), float(d.get("uncertainty_variance", 0.0)),
                        float(d.get("ground_range_m", 0.0)), d.get("channel", "Port"),
                        float(d.get("error_ellipse_a", 0.0)), float(d.get("error_ellipse_b", 0.0)),
                        float(d.get("error_ellipse_phi", 0.0)),
                        lon, lat
                    ))
                conn.commit()
                cur.close()
                conn.close()
                return
            except Exception:
                pass

        # Fallback to local SQLite spatial store
        self.sqlite_db.save_detections(detections, mission_id=mission_id)

    def export_postgis_sql_dump(self, detections: List[Dict[str, Any]], output_path: str = "outputs/postgis_dump.sql") -> str:
        """
        Generates a standalone PostGIS SQL script with DDL and INSERT statements
        ready for direct deployment to any cloud/local PostgreSQL+PostGIS server.
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        lines = [POSTGIS_SCHEMA_SQL, "\n-- Injected Survey Sightings\n"]
        lines.append("INSERT INTO sonar_missions (mission_id, mission_name, vessel_name) VALUES ('mission_sih_2026', 'SIH Marine Survey', 'RV Akhet') ON CONFLICT DO NOTHING;\n")

        for d in detections:
            if "latitude" not in d or "longitude" not in d:
                continue
            lat = d["latitude"]
            lon = d["longitude"]
            cname = d.get("class_name", "Target").replace("'", "''")
            conf = d.get("conf", 0.0)
            fused = d.get("fused_confidence", conf)
            unc = d.get("uncertainty_flag", "LOW")
            var = d.get("uncertainty_variance", 0.0)
            rg = d.get("ground_range_m", 0.0)
            ch = d.get("channel", "Port")
            ea = d.get("error_ellipse_a", 5.0)
            eb = d.get("error_ellipse_b", 5.0)
            phi = d.get("error_ellipse_phi", 0.0)

            sql_insert = (
                f"INSERT INTO marine_debris_detections "
                f"(mission_id, class_name, confidence, fused_confidence, uncertainty_flag, "
                f"uncertainty_variance, ground_range_m, channel, error_ellipse_a, error_ellipse_b, "
                f"error_ellipse_phi, geom) VALUES ("
                f"'mission_sih_2026', '{cname}', {conf:.3f}, {fused:.3f}, '{unc}', "
                f"{var:.4f}, {rg:.2f}, '{ch}', {ea:.2f}, {eb:.2f}, {phi:.1f}, "
                f"ST_SetSRID(ST_MakePoint({lon:.6f}, {lat:.6f}), 4326));"
            )
            lines.append(sql_insert)

        dump_str = "\n".join(lines)
        with open(p, "w", encoding="utf-8") as f:
            f.write(dump_str)

        return dump_str

-- MarineGuard detection store  (PostgreSQL + PostGIS).  SRID 4326 = WGS-84.
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS surveys (
    survey_id      TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    source_file    TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    scqi           DOUBLE PRECISION,
    snr_db         DOUBLE PRECISION,
    config_hash    TEXT,
    model_version  JSONB
);

CREATE TABLE IF NOT EXISTS detections (
    id                      BIGSERIAL PRIMARY KEY,
    survey_id               TEXT NOT NULL REFERENCES surveys(survey_id) ON DELETE CASCADE,
    run_id                  TEXT NOT NULL,
    frame_id                INTEGER NOT NULL,
    "timestamp"             DOUBLE PRECISION,
    class_name              TEXT NOT NULL,
    detection_type          TEXT NOT NULL CHECK (detection_type IN ('known', 'unknown', 'noise')),
    confidence              DOUBLE PRECISION,
    bbox                    JSONB,                 -- [x1, y1, x2, y2] pixels (frame coordinates)
    segmentation_mask       JSONB,                 -- polygon [[x, y], ...] pixels
    mask_area_px            INTEGER,
    anomaly_score           DOUBLE PRECISION,
    reconstruction_error    DOUBLE PRECISION,
    track_id                INTEGER,
    latitude                DOUBLE PRECISION,
    longitude               DOUBLE PRECISION,
    positional_uncertainty_m DOUBLE PRECISION,
    snr_db                  DOUBLE PRECISION,
    scqi                    DOUBLE PRECISION,
    model_version           JSONB NOT NULL,
    config_hash             TEXT NOT NULL,
    evidence                JSONB,
    geom                    geometry(Point, 4326)
);

CREATE INDEX IF NOT EXISTS detections_geom_gix   ON detections USING GIST (geom);
CREATE INDEX IF NOT EXISTS detections_survey_ix  ON detections (survey_id, frame_id);
CREATE INDEX IF NOT EXISTS detections_track_ix   ON detections (survey_id, track_id);
CREATE INDEX IF NOT EXISTS detections_type_ix    ON detections (detection_type);

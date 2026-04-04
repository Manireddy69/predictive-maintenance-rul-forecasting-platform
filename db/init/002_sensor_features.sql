CREATE TABLE IF NOT EXISTS telemetry.sensor_features (
    event_time TIMESTAMPTZ NOT NULL,
    equipment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    cycle BIGINT,
    source_file TEXT,
    feature_set_version TEXT NOT NULL DEFAULT 'day4_v1',
    feature_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (event_time, equipment_id, run_id)
);

SELECT create_hypertable(
    'telemetry.sensor_features',
    'event_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_sensor_features_equipment_time
    ON telemetry.sensor_features (equipment_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_sensor_features_version
    ON telemetry.sensor_features (feature_set_version);

CREATE INDEX IF NOT EXISTS idx_sensor_features_payload_gin
    ON telemetry.sensor_features
    USING GIN (feature_payload);

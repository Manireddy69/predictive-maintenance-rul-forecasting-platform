CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE SCHEMA IF NOT EXISTS telemetry;

CREATE TABLE IF NOT EXISTS telemetry.sensor_readings (
    event_time TIMESTAMPTZ NOT NULL,
    equipment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    cycle BIGINT,
    setting_1 DOUBLE PRECISION,
    setting_2 DOUBLE PRECISION,
    setting_3 DOUBLE PRECISION,
    sensor_1 DOUBLE PRECISION,
    sensor_2 DOUBLE PRECISION,
    sensor_3 DOUBLE PRECISION,
    sensor_4 DOUBLE PRECISION,
    sensor_5 DOUBLE PRECISION,
    sensor_6 DOUBLE PRECISION,
    sensor_7 DOUBLE PRECISION,
    sensor_8 DOUBLE PRECISION,
    sensor_9 DOUBLE PRECISION,
    sensor_10 DOUBLE PRECISION,
    sensor_11 DOUBLE PRECISION,
    sensor_12 DOUBLE PRECISION,
    sensor_13 DOUBLE PRECISION,
    sensor_14 DOUBLE PRECISION,
    sensor_15 DOUBLE PRECISION,
    sensor_16 DOUBLE PRECISION,
    sensor_17 DOUBLE PRECISION,
    sensor_18 DOUBLE PRECISION,
    sensor_19 DOUBLE PRECISION,
    sensor_20 DOUBLE PRECISION,
    sensor_21 DOUBLE PRECISION,
    failure_label SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (event_time, equipment_id, run_id)
);

SELECT create_hypertable(
    'telemetry.sensor_readings',
    'event_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_equipment_time
    ON telemetry.sensor_readings (equipment_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_run_time
    ON telemetry.sensor_readings (run_id, event_time DESC);

ALTER TABLE telemetry.sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'event_time ASC',
    timescaledb.compress_segmentby = 'equipment_id, run_id',
    timescaledb.compress_chunk_time_interval = '30 days'
);

SELECT add_compression_policy(
    'telemetry.sensor_readings',
    compress_after => INTERVAL '7 days',
    if_not_exists => TRUE
);

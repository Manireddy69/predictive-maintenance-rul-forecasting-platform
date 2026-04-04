from __future__ import annotations

import pandas as pd

from src.feature_engineering import engineer_telemetry_features, prepare_telemetry_batch


def _make_sensor_batch() -> pd.DataFrame:
    rows = []
    base_time = pd.Timestamp("2026-04-04T00:00:00Z")
    for cycle in range(1, 5):
        row = {
            "event_time": base_time + pd.Timedelta(minutes=cycle),
            "equipment_id": "engine_001",
            "sensor_1": float(cycle),
            "sensor_2": float(cycle * 2),
        }
        for sensor_index in range(3, 22):
            row[f"sensor_{sensor_index}"] = float(cycle + sensor_index)
        rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_telemetry_batch_derives_run_id_and_cycle() -> None:
    batch_df = _make_sensor_batch().drop(columns=["event_time"]).assign(
        event_time=[
            "2026-04-04T00:01:00Z",
            "2026-04-04T00:02:00Z",
            "2026-04-04T00:03:00Z",
            "2026-04-04T00:04:00Z",
        ]
    )

    prepared_df = prepare_telemetry_batch(batch_df, source_file="sample.csv")

    assert prepared_df["run_id"].nunique() == 1
    assert prepared_df["run_id"].iloc[0] == "engine_001_batch_run_001"
    assert prepared_df["cycle"].tolist() == [1, 2, 3, 4]
    assert prepared_df["source_file"].iloc[0] == "sample.csv"


def test_engineer_telemetry_features_builds_rolling_lag_fft_and_ratio_columns() -> None:
    feature_df = engineer_telemetry_features(_make_sensor_batch())
    last_row = feature_df.iloc[-1]

    assert "sensor_1_1h_mean" in feature_df.columns
    assert "sensor_1_lag_2" in feature_df.columns
    assert "sensor_1_fft_amp_1" in feature_df.columns
    assert "sensor_2_to_sensor_1_ratio" in feature_df.columns
    assert last_row["sensor_1_1h_mean"] == 2.5
    assert last_row["sensor_1_lag_2"] == 2.0
    assert last_row["sensor_2_to_sensor_1_ratio"] == 2.0
    assert pd.notna(last_row["sensor_1_fft_amp_1"])

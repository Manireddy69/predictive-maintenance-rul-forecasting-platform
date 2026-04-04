from __future__ import annotations

import pandas as pd

from src.batch_validation import validate_sensor_batch


def _make_valid_batch() -> pd.DataFrame:
    rows = []
    for cycle in range(1, 4):
        row = {
            "event_time": f"2026-04-04T00:0{cycle}:00Z",
            "equipment_id": "engine_001",
            "run_id": "engine_001_run_001",
        }
        for sensor_index in range(1, 22):
            row[f"sensor_{sensor_index}"] = 500.0 + cycle + sensor_index
        rows.append(row)
    return pd.DataFrame(rows)


def test_validate_sensor_batch_passes_on_valid_input() -> None:
    report = validate_sensor_batch(_make_valid_batch())

    assert report.success is True


def test_validate_sensor_batch_fails_on_duplicate_keys() -> None:
    batch_df = _make_valid_batch()
    duplicated_df = pd.concat([batch_df, batch_df.iloc[[0]]], ignore_index=True)

    report = validate_sensor_batch(duplicated_df)

    assert report.success is False
    assert any(check.name == "unique_batch_keys" and not check.success for check in report.checks)

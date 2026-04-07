from __future__ import annotations

import pandas as pd

from src.kafka_to_timescaledb_consumer import normalize_stream_message


def _make_message() -> dict[str, object]:
    message: dict[str, object] = {
        "event_time": "2026-04-07T00:01:00Z",
        "equipment_id": "engine_001",
        "run_id": "stream_run_001",
        "cycle": 4,
        "setting_1": 0.01,
        "setting_2": 0.001,
        "setting_3": 100.0,
        "failure_label": 0,
    }
    for sensor_index in range(1, 22):
        message[f"sensor_{sensor_index}"] = 500.0 + sensor_index
    return message


def test_normalize_stream_message_coerces_expected_types() -> None:
    normalized = normalize_stream_message(_make_message())

    assert isinstance(normalized["event_time"], pd.Timestamp)
    assert normalized["event_time"].isoformat() == "2026-04-07T00:01:00+00:00"
    assert normalized["cycle"] == 4
    assert normalized["failure_label"] == 0
    assert normalized["sensor_21"] == 521.0

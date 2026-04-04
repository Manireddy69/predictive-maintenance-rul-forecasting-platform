from __future__ import annotations

import json

import pandas as pd

from src.ingest_feature_timescaledb import pack_feature_rows


def test_pack_feature_rows_collects_feature_payload() -> None:
    feature_df = pd.DataFrame(
        [
            {
                "event_time": "2026-04-04T00:01:00Z",
                "equipment_id": "engine_001",
                "run_id": "engine_001_run_001",
                "cycle": 1,
                "source_file": "sensor_batch.csv",
                "feature_set_version": "day4_v1",
                "sensor_1_1h_mean": 10.5,
                "sensor_2_to_sensor_1_ratio": 2.0,
            }
        ]
    )

    packed_rows = pack_feature_rows(feature_df)
    feature_payload = json.loads(packed_rows[0][-1])

    assert len(packed_rows) == 1
    assert packed_rows[0][1] == "engine_001"
    assert feature_payload["sensor_1_1h_mean"] == 10.5
    assert feature_payload["sensor_2_to_sensor_1_ratio"] == 2.0

from __future__ import annotations

import pandas as pd

from src.data import SENSOR_COLUMNS
from src.telemetry import TELEMETRY_COLUMNS, map_cmapss_to_telemetry


def test_map_cmapss_to_telemetry_builds_expected_identifiers_and_time() -> None:
    row = {
        "unit": 2,
        "cycle": 3,
        "setting_1": 0.1,
        "setting_2": 0.2,
        "setting_3": 100.0,
    }
    row.update({column: float(index) for index, column in enumerate(SENSOR_COLUMNS, start=1)})
    df = pd.DataFrame([row])

    telemetry_df = map_cmapss_to_telemetry(
        df,
        fd="FD001",
        split="train",
        cycle_seconds=60,
        start_time="2026-01-01T00:00:00Z",
        failure_label=0,
    )

    assert list(telemetry_df.columns) == TELEMETRY_COLUMNS
    assert telemetry_df.loc[0, "equipment_id"] == "FD001_train_unit_002"
    assert telemetry_df.loc[0, "run_id"] == "FD001_train_unit_002_run_001"
    assert telemetry_df.loc[0, "event_time"].isoformat() == "2026-01-01T00:02:00+00:00"
    assert telemetry_df.loc[0, "sensor_21"] == 21.0

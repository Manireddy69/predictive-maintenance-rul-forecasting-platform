from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from .data import SENSOR_COLUMNS

TELEMETRY_COLUMNS = [
    "event_time",
    "equipment_id",
    "run_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *SENSOR_COLUMNS,
    "failure_label",
]


def _base_timestamp(start_time: str | None = None) -> datetime:
    if start_time is None:
        return datetime(2026, 1, 1, tzinfo=UTC)
    return datetime.fromisoformat(start_time.replace("Z", "+00:00"))


def map_cmapss_to_telemetry(
    df: pd.DataFrame,
    fd: str,
    split: str,
    cycle_seconds: int = 60,
    start_time: str | None = None,
    failure_label: int = 0,
) -> pd.DataFrame:
    """
    Map CMAPSS rows into the telemetry schema used by TimescaleDB.

    CMAPSS does not contain real timestamps, equipment IDs, or run IDs, so we derive
    them deterministically from the available dataset keys.
    """
    required_columns = {"unit", "cycle", "setting_1", "setting_2", "setting_3", *SENSOR_COLUMNS}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Cannot map CMAPSS data without columns: {missing_str}")

    telemetry_df = df.copy()
    start_dt = _base_timestamp(start_time)

    telemetry_df["event_time"] = telemetry_df["cycle"].apply(
        lambda cycle: start_dt + timedelta(seconds=(int(cycle) - 1) * cycle_seconds)
    )
    telemetry_df["equipment_id"] = telemetry_df["unit"].apply(lambda unit: f"{fd}_{split}_unit_{int(unit):03d}")
    telemetry_df["run_id"] = telemetry_df["unit"].apply(lambda unit: f"{fd}_{split}_unit_{int(unit):03d}_run_001")
    telemetry_df["failure_label"] = int(failure_label)

    return telemetry_df[TELEMETRY_COLUMNS]

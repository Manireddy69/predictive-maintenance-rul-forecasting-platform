from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .batch_validation import ValidationReport, validate_sensor_batch
from .feature_engineering import engineer_telemetry_features, prepare_telemetry_batch


def load_sensor_csv(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def stage_sensor_batch_csv(
    csv_path: str | Path,
    staging_dir: str | Path,
) -> Path:
    csv_path = Path(csv_path)
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_sensor_csv(csv_path)
    staged_df = prepare_telemetry_batch(raw_df, source_file=csv_path.name)

    staged_path = staging_dir / "normalized_sensor_batch.csv"
    staged_df.to_csv(staged_path, index=False)
    return staged_path


def validate_staged_batch_csv(
    staged_csv_path: str | Path,
    report_path: str | Path | None = None,
) -> ValidationReport:
    staged_df = pd.read_csv(staged_csv_path)
    validation_report = validate_sensor_batch(staged_df)

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(validation_report.to_dict(), indent=2), encoding="utf-8")

    if not validation_report.success:
        raise ValueError(validation_report.to_error_message())

    return validation_report


def engineer_staged_features(
    staged_csv_path: str | Path,
    output_path: str | Path,
    feature_set_version: str = "day4_v1",
) -> Path:
    staged_df = pd.read_csv(staged_csv_path)
    staged_df["event_time"] = pd.to_datetime(staged_df["event_time"], utc=True, errors="coerce")
    feature_df = engineer_telemetry_features(staged_df)
    feature_df["feature_set_version"] = feature_set_version

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    return output_path

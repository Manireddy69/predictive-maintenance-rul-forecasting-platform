from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .batch_pipeline import engineer_staged_features, stage_sensor_batch_csv, validate_staged_batch_csv
from .ingest_feature_timescaledb import insert_sensor_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Day 4 batch feature pipeline locally.")
    parser.add_argument(
        "--csv-path",
        default="Data/batch/sensor_batch.csv",
        help="Raw sensor CSV path.",
    )
    parser.add_argument(
        "--staging-dir",
        default="Data/batch/staging",
        help="Directory for normalized batch files and validation artifacts.",
    )
    parser.add_argument(
        "--feature-set-version",
        default="day4_v1",
        help="Feature set version stored with the engineered rows.",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Run staging, validation, and feature engineering without loading TimescaleDB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    staging_dir = Path(args.staging_dir)
    feature_path = staging_dir / "engineered_sensor_features.csv"
    report_path = staging_dir / "validation_report.json"

    staged_path = stage_sensor_batch_csv(args.csv_path, staging_dir=staging_dir)
    validation_report = validate_staged_batch_csv(staged_path, report_path=report_path)
    engineered_path = engineer_staged_features(
        staged_csv_path=staged_path,
        output_path=feature_path,
        feature_set_version=args.feature_set_version,
    )

    print(f"Staged CSV: {staged_path}")
    print(f"Validation engine: {validation_report.engine}")
    print(f"Feature CSV: {engineered_path}")

    if args.skip_load:
        return

    feature_df = pd.read_csv(engineered_path)
    inserted = insert_sensor_features(feature_df)
    print(f"Loaded {inserted} engineered feature rows into telemetry.sensor_features.")


if __name__ == "__main__":
    main()

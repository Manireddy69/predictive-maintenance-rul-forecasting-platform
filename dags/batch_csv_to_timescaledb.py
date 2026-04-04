from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # pragma: no cover - imported only in Airflow environments
    DAG = None
    PythonOperator = None

from src.batch_pipeline import engineer_staged_features, stage_sensor_batch_csv, validate_staged_batch_csv
from src.ingest_feature_timescaledb import insert_sensor_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "Data" / "batch" / "sensor_batch.csv"
DEFAULT_STAGING_DIR = PROJECT_ROOT / "Data" / "batch" / "staging"
DEFAULT_FEATURE_PATH = DEFAULT_STAGING_DIR / "engineered_sensor_features.csv"
DEFAULT_REPORT_PATH = DEFAULT_STAGING_DIR / "validation_report.json"


def _csv_path() -> Path:
    return Path(os.getenv("BATCH_SENSOR_CSV_PATH", str(DEFAULT_CSV_PATH)))


def _staging_dir() -> Path:
    return Path(os.getenv("BATCH_PIPELINE_STAGING_DIR", str(DEFAULT_STAGING_DIR)))


def ingest_csv_task() -> str:
    staged_path = stage_sensor_batch_csv(
        csv_path=_csv_path(),
        staging_dir=_staging_dir(),
    )
    return str(staged_path)


def validate_with_gx_task() -> str:
    staged_path = _staging_dir() / "normalized_sensor_batch.csv"
    validate_staged_batch_csv(
        staged_csv_path=staged_path,
        report_path=DEFAULT_REPORT_PATH,
    )
    return str(staged_path)


def engineer_features_task() -> str:
    staged_path = _staging_dir() / "normalized_sensor_batch.csv"
    feature_path = engineer_staged_features(
        staged_csv_path=staged_path,
        output_path=DEFAULT_FEATURE_PATH,
    )
    return str(feature_path)


def load_timescaledb_task() -> int:
    import pandas as pd

    feature_df = pd.read_csv(DEFAULT_FEATURE_PATH)
    return insert_sensor_features(feature_df)


if DAG is not None:
    with DAG(
        dag_id="batch_csv_to_timescaledb",
        description="Batch CSV ingestion -> Great Expectations validation -> feature engineering -> TimescaleDB.",
        start_date=datetime(2026, 4, 4),
        schedule="@daily",
        catchup=False,
        tags=["logicveda", "timescaledb", "batch", "features"],
    ) as dag:
        ingest_csv = PythonOperator(
            task_id="ingest_csv",
            python_callable=ingest_csv_task,
        )

        validate_with_gx = PythonOperator(
            task_id="validate_with_gx",
            python_callable=validate_with_gx_task,
        )

        engineer_features = PythonOperator(
            task_id="engineer_features",
            python_callable=engineer_features_task,
        )

        load_timescaledb = PythonOperator(
            task_id="load_timescaledb",
            python_callable=load_timescaledb_task,
        )

        ingest_csv >> validate_with_gx >> engineer_features >> load_timescaledb
else:  # pragma: no cover - helpful for local import without Airflow
    dag = None

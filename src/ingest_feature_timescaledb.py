from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .ingest_timescaledb import get_connection_string

FEATURE_METADATA_COLUMNS = [
    "event_time",
    "equipment_id",
    "run_id",
    "cycle",
    "source_file",
    "feature_set_version",
]


def _load_psycopg():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "psycopg is not installed. Run `pip install -r requirements.txt` before loading feature rows."
        ) from exc
    return psycopg


def pack_feature_rows(df: pd.DataFrame) -> list[tuple[Any, ...]]:
    required_columns = {"event_time", "equipment_id", "run_id"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Cannot load feature rows without columns: {missing_str}")

    prepared = df.copy()
    prepared["event_time"] = pd.to_datetime(prepared["event_time"], utc=True, errors="coerce")
    if prepared["event_time"].isna().any():
        raise ValueError("event_time contains unparseable values.")

    if "cycle" not in prepared.columns:
        prepared["cycle"] = None
    if "source_file" not in prepared.columns:
        prepared["source_file"] = None
    if "feature_set_version" not in prepared.columns:
        prepared["feature_set_version"] = "day4_v1"

    feature_columns = [
        column
        for column in prepared.columns
        if column not in FEATURE_METADATA_COLUMNS
    ]

    rows: list[tuple[Any, ...]] = []
    for record in prepared.to_dict(orient="records"):
        feature_payload = {
            column: _coerce_feature_value(record[column])
            for column in feature_columns
            if record.get(column) is not None
        }
        rows.append(
            (
                record["event_time"],
                record["equipment_id"],
                record["run_id"],
                record.get("cycle"),
                record.get("source_file"),
                record.get("feature_set_version"),
                json.dumps(feature_payload),
            )
        )

    return rows


def insert_sensor_features(df: pd.DataFrame, batch_size: int = 500) -> int:
    psycopg = _load_psycopg()
    rows = pack_feature_rows(df)
    if not rows:
        return 0

    insert_sql = """
        INSERT INTO telemetry.sensor_features (
            event_time,
            equipment_id,
            run_id,
            cycle,
            source_file,
            feature_set_version,
            feature_payload
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (event_time, equipment_id, run_id) DO UPDATE
        SET
            cycle = EXCLUDED.cycle,
            source_file = EXCLUDED.source_file,
            feature_set_version = EXCLUDED.feature_set_version,
            feature_payload = EXCLUDED.feature_payload
    """

    with psycopg.connect(get_connection_string()) as conn:
        with conn.cursor() as cur:
            for start_idx in range(0, len(rows), batch_size):
                batch = rows[start_idx : start_idx + batch_size]
                cur.executemany(insert_sql, batch)
        conn.commit()

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load engineered feature rows into TimescaleDB.")
    parser.add_argument("--csv-path", required=True, help="CSV file containing engineered feature rows.")
    parser.add_argument("--batch-size", type=int, default=500, help="Insert batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Show packed feature rows without inserting them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_df = pd.read_csv(Path(args.csv_path))
    if args.dry_run:
        packed_rows = pack_feature_rows(feature_df)
        print(f"Prepared {len(packed_rows)} feature rows.")
        if packed_rows:
            print(packed_rows[0])
        return

    inserted = insert_sensor_features(feature_df, batch_size=args.batch_size)
    batch_count = int(math.ceil(inserted / args.batch_size)) if inserted else 0
    print(f"Inserted {inserted} feature rows into telemetry.sensor_features across {batch_count} batch(es).")


def _coerce_feature_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd

from src.data import load_fd_data
from src.telemetry import TELEMETRY_COLUMNS, map_cmapss_to_telemetry


def _load_psycopg():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "psycopg is not installed. Run `pip install -r requirements.txt` before ingesting into TimescaleDB."
        ) from exc
    return psycopg


def get_connection_string() -> str:
    host = os.getenv("PGHOST", "127.0.0.1")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE", "predictive_maintenance")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "postgres")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def insert_sensor_readings(df: pd.DataFrame, batch_size: int = 1000) -> int:
    psycopg = _load_psycopg()
    placeholders = ", ".join(["%s"] * len(TELEMETRY_COLUMNS))
    insert_sql = f"""
        INSERT INTO telemetry.sensor_readings ({", ".join(TELEMETRY_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT (event_time, equipment_id, run_id) DO NOTHING
    """

    rows = list(df.itertuples(index=False, name=None))
    if not rows:
        return 0

    with psycopg.connect(get_connection_string()) as conn:
        with conn.cursor() as cur:
            for start_idx in range(0, len(rows), batch_size):
                batch = rows[start_idx : start_idx + batch_size]
                cur.executemany(insert_sql, batch)
        conn.commit()

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a CMAPSS split into the local TimescaleDB telemetry table.")
    parser.add_argument("--fd", default="FD001", help="Dataset subset, for example FD001.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Which CMAPSS split to load.")
    parser.add_argument("--limit", type=int, default=500, help="Number of rows to load for the first pass.")
    parser.add_argument(
        "--cycle-seconds",
        type=int,
        default=60,
        help="How many real-world seconds one CMAPSS cycle should represent in the synthetic event_time mapping.",
    )
    parser.add_argument(
        "--start-time",
        default="2026-01-01T00:00:00Z",
        help="Base timestamp used to convert CMAPSS cycle numbers into event_time values.",
    )
    parser.add_argument(
        "--failure-label",
        type=int,
        default=0,
        help="Placeholder label for the ingested rows. Keep 0 unless you are loading known failure windows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the mapped telemetry rows without writing anything to the database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(__file__).resolve().parents[1] / "Data"
    source_df = load_fd_data(data_dir, fd=args.fd, split=args.split)
    telemetry_df = map_cmapss_to_telemetry(
        source_df,
        fd=args.fd,
        split=args.split,
        cycle_seconds=args.cycle_seconds,
        start_time=args.start_time,
        failure_label=args.failure_label,
    )

    if args.limit > 0:
        telemetry_df = telemetry_df.head(args.limit)

    print(f"Prepared {len(telemetry_df)} telemetry rows from {args.fd} {args.split}.")
    print(telemetry_df.head(5).to_string(index=False))

    if args.dry_run:
        return

    inserted = insert_sensor_readings(telemetry_df)
    batch_count = int(math.ceil(inserted / 1000)) if inserted else 0
    print(f"Inserted {inserted} rows into telemetry.sensor_readings across {batch_count} batch(es).")


if __name__ == "__main__":
    main()

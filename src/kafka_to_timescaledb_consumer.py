from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from .ingest_timescaledb import insert_sensor_readings
from .telemetry import TELEMETRY_COLUMNS


def _load_kafka_consumer():
    try:
        from kafka import KafkaConsumer
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "kafka-python is not installed. Run `pip install -r requirements.txt` before starting the consumer."
        ) from exc
    return KafkaConsumer


def normalize_stream_message(message: dict[str, Any]) -> dict[str, Any]:
    missing = [column for column in TELEMETRY_COLUMNS if column not in message]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Kafka message is missing telemetry columns: {missing_str}")

    normalized = {column: message[column] for column in TELEMETRY_COLUMNS}
    normalized["event_time"] = pd.to_datetime(normalized["event_time"], utc=True, errors="coerce")
    if pd.isna(normalized["event_time"]):
        raise ValueError("Kafka message contains an invalid event_time value.")

    normalized["cycle"] = int(normalized["cycle"])
    normalized["failure_label"] = int(normalized["failure_label"])
    for column in TELEMETRY_COLUMNS:
        if column.startswith("sensor_") or column.startswith("setting_"):
            normalized[column] = float(normalized[column])
    return normalized


def insert_stream_batch(messages: list[dict[str, Any]]) -> int:
    if not messages:
        return 0
    batch_df = pd.DataFrame([normalize_stream_message(message) for message in messages], columns=TELEMETRY_COLUMNS)
    return insert_sensor_readings(batch_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume Kafka telemetry messages and load them into TimescaleDB.")
    parser.add_argument("--bootstrap-servers", default="127.0.0.1:29092", help="Kafka bootstrap servers.")
    parser.add_argument("--topic", default="raw-sensor-data", help="Kafka topic to consume from.")
    parser.add_argument("--group-id", default="timescaledb-loader", help="Kafka consumer group id.")
    parser.add_argument("--batch-size", type=int, default=100, help="How many messages to buffer before inserting.")
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Stop after this many consumed messages. Use 0 to keep consuming.",
    )
    parser.add_argument(
        "--auto-offset-reset",
        default="earliest",
        choices=["earliest", "latest"],
        help="Where to start consuming if there is no committed offset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    KafkaConsumer = _load_kafka_consumer()
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        auto_offset_reset=args.auto_offset_reset,
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    print(
        f"Consuming Kafka telemetry from {args.topic} via {args.bootstrap_servers} "
        f"and loading it into telemetry.sensor_readings."
    )

    buffered_messages: list[dict[str, Any]] = []
    consumed_count = 0
    try:
        for record in consumer:
            buffered_messages.append(record.value)
            consumed_count += 1

            if len(buffered_messages) >= args.batch_size:
                inserted = insert_stream_batch(buffered_messages)
                print(f"Inserted batch of {inserted} rows into telemetry.sensor_readings.")
                buffered_messages.clear()

            if args.max_messages and consumed_count >= args.max_messages:
                break
    except KeyboardInterrupt:
        print("Stopping Kafka consumer.")
    finally:
        if buffered_messages:
            inserted = insert_stream_batch(buffered_messages)
            print(f"Inserted final batch of {inserted} rows into telemetry.sensor_readings.")
        consumer.close()


if __name__ == "__main__":
    main()

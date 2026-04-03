from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from src.data import SENSOR_COLUMNS


RAW_TOPIC = "raw-sensor-data"


def _load_kafka_producer():
    try:
        from kafka import KafkaProducer
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "kafka-python is not installed. Run `pip install -r requirements.txt` before starting the producer."
        ) from exc
    return KafkaProducer


@dataclass
class EquipmentStreamState:
    equipment_id: str
    run_id: str
    cycle: int
    baseline: np.ndarray
    drift: np.ndarray
    seasonal: np.ndarray
    noise: np.ndarray
    next_anomaly_cycle: int


def _build_state(equipment_index: int, rng: np.random.Generator, min_gap: int, max_gap: int) -> EquipmentStreamState:
    baseline = rng.normal(500.0, 40.0, size=len(SENSOR_COLUMNS))
    drift = rng.normal(0.12, 0.05, size=len(SENSOR_COLUMNS))
    seasonal = rng.uniform(0.4, 3.0, size=len(SENSOR_COLUMNS))
    noise = rng.uniform(0.15, 1.1, size=len(SENSOR_COLUMNS))

    return EquipmentStreamState(
        equipment_id=f"engine_{equipment_index:03d}",
        run_id=f"stream_run_{equipment_index:03d}",
        cycle=0,
        baseline=baseline,
        drift=drift,
        seasonal=seasonal,
        noise=noise,
        next_anomaly_cycle=int(rng.integers(min_gap, max_gap + 1)),
    )


def _build_sensor_values(state: EquipmentStreamState, rng: np.random.Generator) -> np.ndarray:
    cycle = max(1, state.cycle)
    life_fraction = min(cycle / 400.0, 1.0)
    periodic = state.seasonal * np.sin(cycle / 9.0 + np.arange(len(SENSOR_COLUMNS)) / 5.0)
    gradual_wear = state.drift * cycle
    late_life_stress = state.drift * max(0.0, life_fraction - 0.72) * 120.0
    noise = rng.normal(0.0, state.noise)
    return state.baseline + periodic + gradual_wear + late_life_stress + noise


def _inject_anomaly(
    sensor_values: np.ndarray,
    state: EquipmentStreamState,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    anomalous = sensor_values.copy()
    anomaly_type = random.choice(["spike", "drift_jump", "dropout"])

    if anomaly_type == "spike":
        affected = rng.choice(len(SENSOR_COLUMNS), size=4, replace=False)
        anomalous[affected] += rng.normal(25.0, 6.0, size=len(affected))
    elif anomaly_type == "drift_jump":
        anomalous += rng.normal(8.0, 1.8, size=len(SENSOR_COLUMNS))
    else:
        affected = rng.choice(len(SENSOR_COLUMNS), size=3, replace=False)
        anomalous[affected] = anomalous[affected] * rng.uniform(0.02, 0.08, size=len(affected))

    return anomalous, anomaly_type


def _build_message(
    state: EquipmentStreamState,
    rng: np.random.Generator,
    min_gap: int,
    max_gap: int,
) -> dict[str, object]:
    state.cycle += 1
    event_time = datetime.now(tz=UTC).isoformat()
    sensor_values = _build_sensor_values(state, rng)

    anomaly_type: str | None = None
    is_anomaly = state.cycle == state.next_anomaly_cycle
    if is_anomaly:
        sensor_values, anomaly_type = _inject_anomaly(sensor_values, state, rng)
        state.next_anomaly_cycle += int(rng.integers(min_gap, max_gap + 1))

    message: dict[str, object] = {
        "event_time": event_time,
        "equipment_id": state.equipment_id,
        "run_id": state.run_id,
        "cycle": state.cycle,
        "setting_1": float(rng.normal(0.0, 0.006)),
        "setting_2": float(rng.normal(0.0, 0.0004)),
        "setting_3": 100.0,
        "failure_label": int(is_anomaly),
        "stream_status": "anomaly" if is_anomaly else "normal",
        "anomaly_type": anomaly_type,
    }

    for sensor_name, sensor_value in zip(SENSOR_COLUMNS, sensor_values, strict=True):
        message[sensor_name] = float(sensor_value)

    return message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a real-time telemetry stream and publish it to Kafka."
    )
    parser.add_argument("--bootstrap-servers", default="127.0.0.1:29092", help="Kafka bootstrap servers.")
    parser.add_argument("--topic", default=RAW_TOPIC, help="Kafka topic to publish sensor events to.")
    parser.add_argument("--equipment-count", type=int, default=3, help="How many equipment streams to simulate.")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=1.0,
        help="Pause between each publish round so the stream looks real time.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Stop after this many cycles per equipment. Use 0 to stream forever.",
    )
    parser.add_argument(
        "--anomaly-min-gap",
        type=int,
        default=12,
        help="Minimum number of cycles between injected anomalies for one equipment stream.",
    )
    parser.add_argument(
        "--anomaly-max-gap",
        type=int,
        default=28,
        help="Maximum number of cycles between injected anomalies for one equipment stream.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible streams.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.anomaly_min_gap <= 0 or args.anomaly_max_gap < args.anomaly_min_gap:
        raise ValueError("Anomaly gap values must be positive and max-gap must be >= min-gap.")

    KafkaProducer = _load_kafka_producer()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda value: value.encode("utf-8"),
    )

    states = [
        _build_state(
            equipment_index=index,
            rng=rng,
            min_gap=args.anomaly_min_gap,
            max_gap=args.anomaly_max_gap,
        )
        for index in range(1, args.equipment_count + 1)
    ]

    print(
        f"Streaming telemetry for {args.equipment_count} equipment units to "
        f"{args.topic} via {args.bootstrap_servers}."
    )

    try:
        while True:
            for state in states:
                message = _build_message(
                    state=state,
                    rng=rng,
                    min_gap=args.anomaly_min_gap,
                    max_gap=args.anomaly_max_gap,
                )
                producer.send(args.topic, key=state.equipment_id, value=message)
                status = message["stream_status"]
                anomaly_type = message["anomaly_type"] or "none"
                print(
                    f"[{message['event_time']}] {state.equipment_id} cycle={state.cycle} "
                    f"status={status} anomaly_type={anomaly_type}"
                )

            producer.flush()

            if args.max_cycles and all(state.cycle >= args.max_cycles for state in states):
                break

            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("Stopping sensor stream.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()

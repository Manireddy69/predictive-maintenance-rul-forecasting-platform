from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _make_sensor_columns(num_sensors: int) -> list[str]:
    return [f"sensor_{i}" for i in range(1, num_sensors + 1)]


def _build_unit_profile(num_sensors: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baseline = rng.normal(500, 45, size=num_sensors)
    degradation = np.zeros(num_sensors)
    degradation_count = max(6, num_sensors // 3)
    degrading_idx = rng.choice(num_sensors, size=degradation_count, replace=False)
    degradation[degrading_idx] = rng.normal(18.0, 5.0, size=degradation_count)

    seasonal = rng.uniform(0.0, 4.0, size=num_sensors)
    noise = rng.uniform(0.2, 1.2, size=num_sensors)

    constant_count = max(1, num_sensors // 10)
    constant_idx = rng.choice(num_sensors, size=constant_count, replace=False)
    degradation[constant_idx] = 0.0
    seasonal[constant_idx] = 0.0
    noise[constant_idx] = 0.05

    return baseline, degradation, seasonal, noise


def _build_row(
    unit: int,
    cycle: int,
    total_cycles: int,
    baseline: np.ndarray,
    degradation: np.ndarray,
    seasonal: np.ndarray,
    noise_scale: np.ndarray,
    rng: np.random.Generator,
    anomaly_cycles: set[int],
) -> list[float | int]:
    life_fraction = cycle / total_cycles
    settings = [
        float(rng.normal(0, 0.005)),
        float(rng.normal(0, 0.0003)),
        100.0,
    ]

    sensor_values: list[float] = []
    for sensor_idx in range(len(baseline)):
        noise = float(rng.normal(0, noise_scale[sensor_idx]))
        wear = degradation[sensor_idx] * life_fraction
        late_life_stress = degradation[sensor_idx] * max(0.0, life_fraction - 0.75) * 2.5
        periodic = seasonal[sensor_idx] * np.sin(cycle / 10 + sensor_idx / 4)
        value = float(baseline[sensor_idx] + wear + late_life_stress + periodic + noise)

        if cycle in anomaly_cycles and degradation[sensor_idx] > 0:
            value += float(rng.normal(12.0, 3.0))

        sensor_values.append(value)

    return [unit, cycle, *settings, *sensor_values]


def generate_synthetic_turbofan_dataset(
    num_units: int = 40,
    cycles_min: int = 120,
    cycles_max: int = 260,
    num_sensors: int = 21,
    test_fraction: float = 0.25,
    anomaly_fraction: float = 0.05,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic CMAPSS-like dataset.

    Train units are observed until failure.
    Test units are truncated before failure and paired with an external RUL file.
    """
    rng = np.random.default_rng(random_state)
    column_names = ["unit", "cycle", "setting_1", "setting_2", "setting_3", *_make_sensor_columns(num_sensors)]

    test_unit_count = max(1, int(round(num_units * test_fraction)))
    train_unit_count = max(1, num_units - test_unit_count)

    train_records: list[list[float | int]] = []
    test_records: list[list[float | int]] = []
    rul_records: list[int] = []

    for unit in range(1, train_unit_count + 1):
        total_cycles = int(rng.integers(cycles_min, cycles_max + 1))
        baseline, degradation, seasonal, noise_scale = _build_unit_profile(num_sensors, rng)

        for cycle in range(1, total_cycles + 1):
            train_records.append(
                _build_row(
                    unit=unit,
                    cycle=cycle,
                    total_cycles=total_cycles,
                    baseline=baseline,
                    degradation=degradation,
                    seasonal=seasonal,
                    noise_scale=noise_scale,
                    rng=rng,
                    anomaly_cycles=set(),
                )
            )

    for offset in range(1, test_unit_count + 1):
        unit = offset
        total_cycles = int(rng.integers(cycles_min, cycles_max + 1))
        observed_fraction = float(rng.uniform(0.55, 0.9))
        observed_cycles = max(30, int(total_cycles * observed_fraction))
        rul = total_cycles - observed_cycles
        anomaly_count = max(1, int(observed_cycles * anomaly_fraction))
        anomaly_candidates = range(max(5, observed_cycles - anomaly_count * 2), observed_cycles + 1)
        anomaly_cycles = set(rng.choice(list(anomaly_candidates), size=anomaly_count, replace=False))

        baseline, degradation, seasonal, noise_scale = _build_unit_profile(num_sensors, rng)

        for cycle in range(1, observed_cycles + 1):
            test_records.append(
                _build_row(
                    unit=unit,
                    cycle=cycle,
                    total_cycles=total_cycles,
                    baseline=baseline,
                    degradation=degradation,
                    seasonal=seasonal,
                    noise_scale=noise_scale,
                    rng=rng,
                    anomaly_cycles=anomaly_cycles,
                )
            )

        rul_records.append(rul)

    train_df = pd.DataFrame(train_records, columns=column_names)
    test_df = pd.DataFrame(test_records, columns=column_names)
    rul_df = pd.DataFrame({"rul": rul_records})
    rul_df.index = pd.Index(range(1, len(rul_records) + 1), name="unit")

    return train_df, test_df, rul_df


def save_synthetic_dataset(
    data_dir: Path,
    fd_name: str = "SYNTH",
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    rul_df: pd.DataFrame | None = None,
) -> None:
    """Save a generated synthetic dataset to the project data directory."""
    if train_df is None or test_df is None or rul_df is None:
        raise ValueError("train_df, test_df, and rul_df must all be provided.")

    synthetic_dir = Path(data_dir) / "Synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(synthetic_dir / f"train_{fd_name}.csv", index=False)
    test_df.to_csv(synthetic_dir / f"test_{fd_name}.csv", index=False)
    rul_df.to_csv(synthetic_dir / f"RUL_{fd_name}.csv")

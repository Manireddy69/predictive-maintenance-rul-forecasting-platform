from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SENSOR_COLUMNS

ROLLING_WINDOWS = ("1h", "8h", "24h")
ROLLING_STATS = ("mean", "std", "min", "max")
DEFAULT_LAG_STEPS = tuple(range(1, 13))
DEFAULT_FFT_TOP_K = 5


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    sensor_columns: tuple[str, ...] = tuple(SENSOR_COLUMNS)
    rolling_windows: tuple[str, ...] = ROLLING_WINDOWS
    lag_steps: tuple[int, ...] = DEFAULT_LAG_STEPS
    fft_top_k: int = DEFAULT_FFT_TOP_K


def build_default_ratio_pairs(sensor_columns: list[str] | tuple[str, ...] | None = None) -> list[tuple[str, str]]:
    sensor_columns = list(sensor_columns or SENSOR_COLUMNS)
    return [
        (sensor_columns[index + 1], sensor_columns[index])
        for index in range(len(sensor_columns) - 1)
    ]


def prepare_telemetry_batch(
    df: pd.DataFrame,
    sensor_columns: list[str] | tuple[str, ...] | None = None,
    source_file: str | None = None,
) -> pd.DataFrame:
    sensor_columns = list(sensor_columns or SENSOR_COLUMNS)
    required_columns = {"event_time", "equipment_id", *sensor_columns}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Cannot prepare telemetry batch without columns: {missing_str}")

    prepared = df.copy()
    prepared["event_time"] = pd.to_datetime(prepared["event_time"], utc=True, errors="coerce")
    if prepared["event_time"].isna().any():
        raise ValueError("event_time contains unparseable values.")

    if "run_id" not in prepared.columns:
        prepared["run_id"] = prepared["equipment_id"].astype(str) + "_batch_run_001"

    prepared = prepared.sort_values(["equipment_id", "event_time"]).reset_index(drop=True)

    if "cycle" not in prepared.columns:
        prepared["cycle"] = prepared.groupby("equipment_id").cumcount() + 1

    if source_file is not None and "source_file" not in prepared.columns:
        prepared["source_file"] = source_file

    numeric_columns = [column for column in sensor_columns if column in prepared.columns]
    prepared = prepared.astype({column: float for column in numeric_columns}, copy=False)
    return prepared


def engineer_telemetry_features(
    df: pd.DataFrame,
    config: FeatureEngineeringConfig | None = None,
    ratio_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    config = config or FeatureEngineeringConfig()
    prepared = prepare_telemetry_batch(df, sensor_columns=list(config.sensor_columns))
    ratio_pairs = ratio_pairs or build_default_ratio_pairs(list(config.sensor_columns))

    feature_frames: list[pd.DataFrame] = []
    for _, equipment_df in prepared.groupby("equipment_id", sort=False):
        feature_frames.append(_engineer_equipment_features(equipment_df, config=config, ratio_pairs=ratio_pairs))

    if not feature_frames:
        return prepared.loc[:, ["event_time", "equipment_id", "run_id", "cycle"]].copy()

    return pd.concat(feature_frames, ignore_index=True)


def _engineer_equipment_features(
    equipment_df: pd.DataFrame,
    config: FeatureEngineeringConfig,
    ratio_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    indexed = equipment_df.sort_values("event_time").set_index("event_time")
    base_df = indexed.loc[:, ["equipment_id", "run_id", "cycle"]].copy()

    if "source_file" in indexed.columns:
        base_df["source_file"] = indexed["source_file"]

    engineered_columns: dict[str, np.ndarray | pd.Series] = {}

    for sensor_name in config.sensor_columns:
        sensor_series = indexed[sensor_name].astype(float)

        for rolling_window in config.rolling_windows:
            rolling = sensor_series.rolling(rolling_window, min_periods=1)
            engineered_columns[f"{sensor_name}_{rolling_window}_mean"] = rolling.mean().to_numpy()
            engineered_columns[f"{sensor_name}_{rolling_window}_std"] = rolling.std(ddof=0).fillna(0.0).to_numpy()
            engineered_columns[f"{sensor_name}_{rolling_window}_min"] = rolling.min().to_numpy()
            engineered_columns[f"{sensor_name}_{rolling_window}_max"] = rolling.max().to_numpy()

        for lag_step in config.lag_steps:
            engineered_columns[f"{sensor_name}_lag_{lag_step}"] = sensor_series.shift(lag_step).to_numpy()

        fft_values = _rolling_fft_top_k(
            sensor_series=sensor_series,
            window=config.rolling_windows[-1],
            top_k=config.fft_top_k,
        )
        for index in range(config.fft_top_k):
            engineered_columns[f"{sensor_name}_fft_amp_{index + 1}"] = fft_values[:, index]

    for numerator, denominator in ratio_pairs:
        denominator_values = indexed[denominator].replace(0.0, np.nan)
        engineered_columns[f"{numerator}_to_{denominator}_ratio"] = (
            indexed[numerator] / denominator_values
        ).to_numpy()

    feature_df = pd.concat([base_df, pd.DataFrame(engineered_columns, index=indexed.index)], axis=1)
    return feature_df.reset_index()


def _rolling_fft_top_k(sensor_series: pd.Series, window: str, top_k: int) -> np.ndarray:
    timestamps = sensor_series.index
    values = sensor_series.to_numpy(dtype=float, copy=False)
    results = np.full((len(values), top_k), np.nan, dtype=float)
    window_delta = pd.Timedelta(window)

    start_idx = 0
    for end_idx in range(len(values)):
        while timestamps[end_idx] - timestamps[start_idx] > window_delta:
            start_idx += 1

        window_values = values[start_idx : end_idx + 1]
        if len(window_values) < 2:
            continue

        centered = window_values - np.mean(window_values)
        amplitudes = np.abs(np.fft.rfft(centered))[1:]
        if amplitudes.size == 0:
            continue

        top_values = np.sort(amplitudes)[-top_k:][::-1]
        results[end_idx, : len(top_values)] = top_values

    return results

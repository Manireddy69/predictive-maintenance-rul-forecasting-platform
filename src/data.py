from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from .synthetic import generate_synthetic_turbofan_dataset

OPERATIONAL_SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = ["unit", "cycle", *OPERATIONAL_SETTING_COLUMNS, *SENSOR_COLUMNS]

SplitName = Literal["train", "test"]
SourceName = Literal["nasa", "kaggle", "synthetic"]


def _build_file_path(data_dir: Path, fd: str, split: SplitName) -> Path:
    return Path(data_dir) / "CMaps" / f"{split}_{fd}.txt"


def _build_rul_path(data_dir: Path, fd: str) -> Path:
    return Path(data_dir) / "CMaps" / f"RUL_{fd}.txt"


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("sensor_")]


def load_fd_data(data_dir: Path, fd: str = "FD001", split: SplitName = "train") -> pd.DataFrame:
    """Load one CMAPSS FD00X split with the correct 26-column schema."""
    path = _build_file_path(Path(data_dir), fd, split)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"Unexpected column count for {path.name}: expected {len(CMAPSS_COLUMNS)}, got {df.shape[1]}"
        )

    df.columns = CMAPSS_COLUMNS
    return df


def load_fd_rul(data_dir: Path, fd: str = "FD001") -> pd.DataFrame:
    """Load the CMAPSS test RUL targets."""
    path = _build_rul_path(Path(data_dir), fd)
    if not path.exists():
        raise FileNotFoundError(f"Could not find RUL file: {path}")

    rul_df = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])
    rul_df.index += 1
    rul_df.index.name = "unit"
    return rul_df


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Derive train-set RUL from each unit's final failure cycle."""
    required = {"unit", "cycle"}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Cannot derive RUL without columns: {missing_str}")

    max_cycle = df.groupby("unit")["cycle"].transform("max")
    result = df.copy()
    result["rul"] = max_cycle - result["cycle"]
    return result


def load_kaggle_data(data_dir: Path, dataset_name: str = "predictive_maintenance") -> pd.DataFrame:
    """Load a Kaggle dataset from the project `Data/Kaggle` directory."""
    path = Path(data_dir) / "Kaggle" / f"{dataset_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find Kaggle dataset file: {path}")

    return pd.read_csv(path)


def load_dataset(
    data_dir: Path,
    source: SourceName = "nasa",
    fd: str = "FD001",
    kaggle_name: str = "predictive_maintenance",
    synthetic_units: int = 40,
    synthetic_test_fraction: float = 0.25,
    synthetic_anomaly_fraction: float = 0.05,
    synthetic_random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a dataset from NASA, Kaggle, or generate a synthetic CMAPSS-like dataset."""
    if source == "nasa":
        train_df = load_fd_data(data_dir, fd=fd, split="train")
        test_df = load_fd_data(data_dir, fd=fd, split="test")
        rul_df = load_fd_rul(data_dir, fd=fd)
        return train_df, test_df, rul_df

    if source == "kaggle":
        kaggle_df = load_kaggle_data(data_dir, dataset_name=kaggle_name)
        return kaggle_df, pd.DataFrame(), pd.DataFrame()

    if source == "synthetic":
        return generate_synthetic_turbofan_dataset(
            num_units=synthetic_units,
            cycles_min=120,
            cycles_max=260,
            num_sensors=len(SENSOR_COLUMNS),
            test_fraction=synthetic_test_fraction,
            anomaly_fraction=synthetic_anomaly_fraction,
            random_state=synthetic_random_state,
        )

    raise ValueError(f"Unknown source: {source}")


def summarize_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unit-level summary and numeric summary stats for quick inspection."""
    summary_by_unit = df.groupby("unit").agg(
        cycles=("cycle", "max"),
        unique_settings_1=("setting_1", "nunique"),
        unique_settings_2=("setting_2", "nunique"),
        unique_settings_3=("setting_3", "nunique"),
    )
    summary_stats = df.describe(include="all")
    return summary_by_unit, summary_stats

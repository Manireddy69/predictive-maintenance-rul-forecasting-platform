from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import add_train_rul, get_sensor_columns, load_fd_data, load_fd_rul, summarize_dataset


def summarize_cycles(df: pd.DataFrame) -> dict[str, int]:
    cycle_max = df.groupby("unit")["cycle"].max()
    return {
        "units": int(df["unit"].nunique()),
        "min_cycles": int(cycle_max.min()),
        "median_cycles": int(cycle_max.median()),
        "max_cycles": int(cycle_max.max()),
    }


def get_constant_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    constant_columns: list[str] = []
    for column in df.columns:
        if column in exclude:
            continue
        if df[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)
    return constant_columns


def rank_sensor_variability(df: pd.DataFrame) -> pd.DataFrame:
    sensor_columns = get_sensor_columns(df)
    variability = (
        df[sensor_columns]
        .agg(["std", "nunique"])
        .transpose()
        .sort_values(["std", "nunique"], ascending=False)
    )
    return variability


def estimate_sensor_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """Use start-vs-end sensor deltas as a rough Day 1 degradation signal check."""
    sensor_columns = get_sensor_columns(df)
    first_cycle = df.sort_values(["unit", "cycle"]).groupby("unit").first()
    last_cycle = df.sort_values(["unit", "cycle"]).groupby("unit").last()

    deltas = (last_cycle[sensor_columns] - first_cycle[sensor_columns]).mean().sort_values(key=lambda s: s.abs(), ascending=False)
    return deltas.rename("avg_end_minus_start").to_frame()


def print_dataset_summary(data_dir: Path, fd: str = "FD001") -> None:
    print(f"Loading {fd} data from: {data_dir / 'CMaps'}")

    train_df = load_fd_data(data_dir, fd=fd, split="train")
    test_df = load_fd_data(data_dir, fd=fd, split="test")
    rul_df = load_fd_rul(data_dir, fd=fd)
    train_rul_df = add_train_rul(train_df)

    print("\n=== Dataset overview ===")
    print("Train:", summarize_cycles(train_df))
    print("Test:", summarize_cycles(test_df))
    print(f"Test RUL rows: {len(rul_df)}")

    train_by_unit, train_stats = summarize_dataset(train_df)
    print("\n=== Train summary by unit ===")
    print(train_by_unit.head().to_string())

    print("\n=== Constant columns ===")
    print(get_constant_columns(train_df, exclude={"unit", "cycle"}) or ["None"])

    print("\n=== Most variable sensors ===")
    print(rank_sensor_variability(train_df).head(8).to_string())

    print("\n=== Strongest average degradation deltas ===")
    print(estimate_sensor_degradation(train_df).head(8).to_string())

    print("\n=== Sensor statistics ===")
    sensor_columns = get_sensor_columns(train_df)
    print(train_stats.loc[["mean", "std", "min", "max"], sensor_columns[:8]].transpose().to_string())

    print("\n=== Derived train RUL sample ===")
    print(train_rul_df.loc[train_rul_df["unit"] == 1, ["unit", "cycle", "rul"]].head(5).to_string(index=False))


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "Data"
    print_dataset_summary(data_dir, fd="FD001")

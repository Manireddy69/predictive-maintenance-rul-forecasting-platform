from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import add_train_rul, get_sensor_columns, load_dataset
from src.eda import estimate_sensor_degradation, get_constant_columns, rank_sensor_variability, summarize_cycles


def _print_source_context(source: str, fd: str, kaggle_name: str) -> None:
    print("=== Source role ===")
    if source == "nasa":
        print(f"{fd} is the main dataset for Day 1 because it is a true run-to-failure RUL problem.")
        print("The real job today is to understand unit trajectories, sensor usefulness, and how targets are constructed.")
    elif source == "synthetic":
        print("Synthetic data is optional support material.")
        print("Use it later for controlled anomaly tests, not as proof that the core RUL problem is solved.")
    else:
        print(f"{kaggle_name} is auxiliary data.")
        print("Do not force CMAPSS-style RUL logic onto it unless the schema genuinely supports that.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lean Day 1 EDA for Project-1.")
    parser.add_argument("--source", default="nasa", choices=["nasa", "kaggle", "synthetic"], help="Dataset source to inspect.")
    parser.add_argument("--fd", default="FD001", help="NASA FD dataset variant to load (FD001, FD002, FD003, FD004)")
    parser.add_argument("--kaggle-name", default="predictive_maintenance", help="Kaggle dataset filename (without extension) to load from Data/Kaggle.")
    parser.add_argument("--synthetic-units", type=int, default=40, help="Total synthetic units to generate across train and test.")
    parser.add_argument("--synthetic-test-fraction", type=float, default=0.25, help="Fraction of synthetic units reserved for the test split.")
    parser.add_argument("--synthetic-anomaly-fraction", type=float, default=0.05, help="Fraction of late observed synthetic test cycles that receive anomaly spikes.")
    parser.add_argument("--synthetic-random-state", type=int, default=42, help="Random seed for synthetic data generation.")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "Data"
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Source: {args.source}")
    print(f"Data directory: {data_dir}\n")
    _print_source_context(args.source, args.fd, args.kaggle_name)

    train_df, test_df, rul_df = load_dataset(
        data_dir,
        source=args.source,
        fd=args.fd,
        kaggle_name=args.kaggle_name,
        synthetic_units=args.synthetic_units,
        synthetic_test_fraction=args.synthetic_test_fraction,
        synthetic_anomaly_fraction=args.synthetic_anomaly_fraction,
        synthetic_random_state=args.synthetic_random_state,
    )

    print("=== Dataset shapes ===")
    print(f"train: {train_df.shape}")
    print(f"test:  {test_df.shape}")
    print(f"rul:   {rul_df.shape}\n")

    if "unit" not in train_df.columns or "cycle" not in train_df.columns:
        print("This dataset does not follow the CMAPSS sequence structure.")
        print("Use Day 1 to define what the target and grouping key should be before going further.")
        return

    sensor_columns = get_sensor_columns(train_df)
    print("=== Sequence structure ===")
    print("Train:", summarize_cycles(train_df))
    if not test_df.empty and {"unit", "cycle"}.issubset(test_df.columns):
        print("Test: ", summarize_cycles(test_df))
    if not rul_df.empty:
        print(f"RUL rows: {len(rul_df)}")
    print(f"Sensor columns: {len(sensor_columns)}")
    print()

    print("=== Constant columns ===")
    constant_columns = get_constant_columns(train_df, exclude={"unit", "cycle"})
    print(constant_columns if constant_columns else "No constant columns detected")
    print()

    print("=== Most variable sensors ===")
    variability = rank_sensor_variability(train_df)
    print(variability.head(8).to_string())
    print()

    print("=== Strongest average start-to-end sensor changes ===")
    degradation = estimate_sensor_degradation(train_df)
    print(degradation.head(8).to_string())
    print()

    if args.source in {"nasa", "synthetic"}:
        train_rul_df = add_train_rul(train_df)
        sample_unit = int(train_rul_df["unit"].min())

        print("=== Train RUL sanity check ===")
        print("RUL should decrease toward zero as the unit approaches failure.")
        print(train_rul_df.loc[train_rul_df["unit"] == sample_unit, ["unit", "cycle", "rul"]].head(8).to_string(index=False))
        print()

        top_sensor = variability.index[0]
        print("=== Day 1 takeaways ===")
        print(f"Most informative early sensor candidate: {top_sensor}")
        print("Before modeling, decide which sensors to keep, how to split by unit/time, and how to avoid leakage.")


if __name__ == "__main__":
    main()

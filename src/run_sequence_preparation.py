from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_dataset
from .sequence_data import CMAPSSSequenceDataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CMAPSS sliding-window datasets and inspect Lightning DataModule batches."
    )
    parser.add_argument("--fd", type=str, default="FD001", help="CMAPSS subset to load, for example FD001.")
    parser.add_argument(
        "--target-mode",
        type=str,
        default="rul",
        choices=["rul", "failure_in_next_window"],
        help="Sequence target type.",
    )
    parser.add_argument("--window-size", type=int, default=30, help="Sliding-window length.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride.")
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=30,
        help="Future-cycle horizon used for `failure_in_next_window` targets.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for DataLoaders.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of train units reserved for validation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Use a positive value for faster batching on larger runs.",
    )
    return parser.parse_args()


def _print_split_summary(split_name: str, metadata) -> None:
    if metadata is None or metadata.empty:
        print(f"{split_name}: no windows generated")
        return

    line = (
        f"{split_name}: windows={len(metadata)}, "
        f"units={metadata['unit'].nunique()}, "
        f"cycle_range=({int(metadata['cycle_start'].min())}, {int(metadata['cycle_end'].max())})"
    )
    print(line)

    if "failure_in_next_window" in metadata.columns:
        positives = int(metadata["failure_in_next_window"].sum())
        print(f"{split_name}: positive_windows={positives}, negative_windows={len(metadata) - positives}")
    elif "rul" in metadata.columns:
        print(
            f"{split_name}: rul_range=({float(metadata['rul'].min()):.1f}, {float(metadata['rul'].max()):.1f})"
        )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "Data"

    train_df, test_df, rul_df = load_dataset(data_dir=data_dir, source="nasa", fd=args.fd)

    datamodule = CMAPSSSequenceDataModule(
        train_df=train_df,
        test_df=test_df,
        test_rul_df=rul_df,
        target_mode=args.target_mode,
        window_size=args.window_size,
        stride=args.stride,
        prediction_horizon=args.prediction_horizon,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup()

    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    test_batch = next(iter(datamodule.test_dataloader()))

    print("=== Sequence Preparation Summary ===")
    print(f"FD subset: {args.fd}")
    print(f"Target mode: {args.target_mode}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Prediction horizon: {args.prediction_horizon}")
    print(f"Target column: {datamodule.target_name_}")
    print(f"Feature count: {len(datamodule.feature_columns_ or [])}")
    print()

    _print_split_summary("train", datamodule.train_metadata)
    _print_split_summary("val", datamodule.val_metadata)
    _print_split_summary("test", datamodule.test_metadata)
    print()

    print("=== Batch Shapes ===")
    print(f"train batch: x={tuple(train_batch[0].shape)}, y={tuple(train_batch[1].shape)}")
    print(f"val batch:   x={tuple(val_batch[0].shape)}, y={tuple(val_batch[1].shape)}")
    print(f"test batch:  x={tuple(test_batch[0].shape)}, y={tuple(test_batch[1].shape)}")
    print()

    print("=== Metadata Preview ===")
    print(datamodule.train_metadata.head().to_string(index=False))


if __name__ == "__main__":
    main()

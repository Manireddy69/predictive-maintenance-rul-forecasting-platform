from __future__ import annotations

import argparse
import json

from .sequence_attention_model import (
    OptunaSearchConfig,
    SequenceModelConfig,
    SequenceTrainingConfig,
    train_day9_sequence_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the day 9 bidirectional LSTM + attention sequence model with optional Optuna tuning."
    )
    parser.add_argument("--fd", type=str, default="FD001", help="CMAPSS subset to load, for example FD001.")
    parser.add_argument(
        "--target-mode",
        type=str,
        default="rul",
        choices=["rul", "failure_in_next_window"],
        help="Supervised target type for sequence modeling.",
    )
    parser.add_argument("--window-size", type=int, default=30, help="Base sliding-window length.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride.")
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=30,
        help="Future-cycle horizon for `failure_in_next_window` targets.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--max-epochs", type=int, default=15, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=4, help="Early-stopping patience.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of train units reserved for validation.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Base LSTM hidden size.")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of bidirectional LSTM layers.",
    )
    parser.add_argument("--dropout", type=float, default=0.25, help="Base dropout rate.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Base learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=10,
        help="Number of Optuna trials. Use 0 to skip tuning and train only the base configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Data/experiments/day9_sequence_training",
        help="Directory where day 9 artifacts will be saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional folder name for the saved run. Defaults to a UTC timestamp.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_config = SequenceModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    training_config = SequenceTrainingConfig(
        target_mode=args.target_mode,
        window_size=args.window_size,
        stride=args.stride,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        validation_fraction=args.validation_fraction,
        num_workers=args.num_workers,
        random_state=args.random_state,
    )
    search_config = OptunaSearchConfig(n_trials=args.optuna_trials)

    metrics, _, artifacts = train_day9_sequence_model(
        fd=args.fd,
        model_config=model_config,
        training_config=training_config,
        search_config=search_config,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )

    print("=== Day 9 sequence model results ===")
    print(json.dumps(metrics, indent=2))
    print()
    print(f"Artifacts: {artifacts.output_dir}")
    print(f"Predictions: {artifacts.predictions_csv}")
    print(f"Summary: {artifacts.summary_markdown}")
    if artifacts.trials_csv is not None:
        print(f"Optuna trials: {artifacts.trials_csv}")
    if artifacts.checkpoint_path is not None:
        print(f"Best checkpoint: {artifacts.checkpoint_path}")


if __name__ == "__main__":
    main()

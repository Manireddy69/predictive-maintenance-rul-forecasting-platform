from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_predictions_path() -> Path:
    return project_root() / "Data" / "experiments" / "day9_sequence_training" / "fd001_live_check" / "test_predictions.csv"


def evaluate_rul_predictions(predictions_csv: Path) -> dict[str, float | int]:
    predictions = pd.read_csv(predictions_csv)
    required_columns = {"rul", "prediction"}
    missing = required_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required RUL prediction columns: {', '.join(sorted(missing))}")

    y_true = predictions["rul"].to_numpy(dtype=float)
    y_pred = predictions["prediction"].to_numpy(dtype=float)
    nonzero_mask = np.abs(y_true) > 1e-9
    if not nonzero_mask.any():
        raise ValueError("Cannot compute MAPE because all RUL targets are zero.")

    absolute_error = np.abs(y_true - y_pred)
    percentage_error = absolute_error[nonzero_mask] / np.abs(y_true[nonzero_mask])
    within_10_cycles = absolute_error <= 10
    within_20_cycles = absolute_error <= 20

    return {
        "row_count": int(len(predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(percentage_error)),
        "mape_percent": float(np.mean(percentage_error) * 100.0),
        "median_absolute_error": float(np.median(absolute_error)),
        "within_10_cycles_rate": float(np.mean(within_10_cycles)),
        "within_20_cycles_rate": float(np.mean(within_20_cycles)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved RUL predictions against capstone acceptance metrics.")
    parser.add_argument("--predictions-csv", type=Path, default=default_predictions_path(), help="Saved RUL predictions CSV.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root() / "reports" / "rul_acceptance_metrics.json",
        help="Path for JSON metrics evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_rul_predictions(args.predictions_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=== RUL acceptance metrics ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nJSON: {args.output_json}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


SCORE_COLUMNS = {
    "lstm_autoencoder": "reconstruction_error",
    "isolation_forest": "isolation_forest_score",
    "local_outlier_factor": "local_outlier_factor_score",
    "zscore": "zscore_score",
    "mad": "mad_score",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_scores_path() -> Path:
    return project_root() / "Data" / "experiments" / "anomaly_day6" / "day6_final" / "window_scores.csv"


def _best_threshold_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        raise ValueError("At least one score threshold is required for anomaly evaluation.")

    threshold_precision = precision[1:]
    threshold_recall = recall[1:]
    f1 = (2 * threshold_precision * threshold_recall) / np.maximum(threshold_precision + threshold_recall, 1e-12)
    best_index = int(np.nanargmax(f1))
    best_threshold = float(thresholds[best_index])
    y_pred = (scores >= best_threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    false_alarm_rate = fp / max(fp + tn, 1)
    return {
        "threshold": best_threshold,
        "precision": float(threshold_precision[best_index]),
        "recall": float(threshold_recall[best_index]),
        "f1": float(f1[best_index]),
        "false_alarm_rate": float(false_alarm_rate),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_anomaly_file(scores_csv: Path) -> pd.DataFrame:
    scores_frame = pd.read_csv(scores_csv)
    if "is_anomaly_window" not in scores_frame.columns:
        raise ValueError("Expected `is_anomaly_window` column in anomaly score file.")

    y_true = scores_frame["is_anomaly_window"].to_numpy(dtype=int)
    rows: list[dict[str, float | str | int]] = []
    for model_key, score_column in SCORE_COLUMNS.items():
        if score_column not in scores_frame.columns:
            continue
        scores = scores_frame[score_column].to_numpy(dtype=float)
        metrics = _best_threshold_metrics(y_true, scores)
        rows.append(
            {
                "model_key": model_key,
                "score_column": score_column,
                "roc_auc": float(roc_auc_score(y_true, scores)),
                "pr_auc": float(average_precision_score(y_true, scores)),
                **metrics,
            }
        )

    if not rows:
        raise ValueError("No supported anomaly score columns were found.")
    return pd.DataFrame(rows).sort_values(["f1", "pr_auc"], ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved anomaly scores against LogicVeda acceptance metrics.")
    parser.add_argument("--scores-csv", type=Path, default=default_scores_path(), help="Saved anomaly window scores CSV.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root() / "reports" / "anomaly_acceptance_metrics.json",
        help="Path for JSON metrics evidence.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root() / "reports" / "anomaly_acceptance_metrics.csv",
        help="Path for CSV metrics evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_frame = evaluate_anomaly_file(args.scores_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(metrics_frame.to_dict(orient="records"), indent=2), encoding="utf-8")

    print("=== Anomaly acceptance metrics ===")
    print(metrics_frame.to_string(index=False))
    print(f"\nCSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()

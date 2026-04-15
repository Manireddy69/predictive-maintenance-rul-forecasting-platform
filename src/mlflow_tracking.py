from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "MLflow is not installed. Run `pip install mlflow` or add it to your environment before logging runs."
        ) from exc
    return mlflow


def default_tracking_uri(project_root: Path) -> str:
    return (Path(project_root) / "mlruns").resolve().as_uri()


def compute_regression_calibration_metrics(
    targets: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    num_bins: int = 10,
) -> dict[str, float]:
    targets_array = np.asarray(targets, dtype=float)
    predictions_array = np.asarray(predictions, dtype=float)
    if targets_array.shape != predictions_array.shape:
        raise ValueError("targets and predictions must have the same shape.")
    if targets_array.size < 2:
        return {
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_mae": float("nan"),
        }

    design = np.vstack([predictions_array, np.ones_like(predictions_array)]).T
    slope, intercept = np.linalg.lstsq(design, targets_array, rcond=None)[0]

    bin_count = max(2, min(int(num_bins), int(targets_array.size)))
    bin_frame = pd.DataFrame({"target": targets_array, "prediction": predictions_array})
    bin_frame["bin"] = pd.qcut(bin_frame["prediction"], q=bin_count, duplicates="drop")
    grouped = bin_frame.groupby("bin", observed=False).agg(target_mean=("target", "mean"), prediction_mean=("prediction", "mean"))
    calibration_mae = float(np.abs(grouped["target_mean"] - grouped["prediction_mean"]).mean()) if not grouped.empty else float("nan")

    return {
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "calibration_mae": calibration_mae,
    }


def log_week1_anomaly_checkpoint(
    comparison_df: pd.DataFrame,
    metadata: dict[str, Any],
    artifact_paths: list[Path],
    project_root: Path,
    experiment_name: str = "week1-anomaly-checkpoint",
    run_name: str | None = None,
    tracking_uri: str | None = None,
) -> str:
    mlflow = _load_mlflow()
    tracking_uri = tracking_uri or default_tracking_uri(project_root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = run_name or str(metadata.get("run_name", "week1_checkpoint"))
    with mlflow.start_run(run_name=run_name) as run:
        best_model = metadata.get("best_model", {})
        config = metadata.get("config", {})

        mlflow.log_param("run_name", metadata.get("run_name"))
        mlflow.log_param("device_name", metadata.get("device_name"))
        mlflow.log_param("best_model_key", best_model.get("model_key"))
        mlflow.log_param("best_model_name", best_model.get("model_name"))
        mlflow.log_param("anomaly_fraction", metadata.get("anomaly_fraction"))
        mlflow.log_param("random_state", metadata.get("random_state"))

        for key, value in config.items():
            mlflow.log_param(f"config_{key}", value)

        if "threshold" in metadata:
            mlflow.log_metric("reconstruction_threshold", float(metadata["threshold"]))

        for row in comparison_df.to_dict(orient="records"):
            model_key = str(row["model_key"])
            mlflow.log_metric(f"{model_key}_roc_auc", float(row["roc_auc"]))
            mlflow.log_metric(f"{model_key}_pr_auc", float(row["pr_auc"]))
            mlflow.log_metric(f"{model_key}_pr_curve_points", float(row["pr_curve_points"]))

        best_row = comparison_df.iloc[0]
        mlflow.log_metric("best_roc_auc", float(best_row["roc_auc"]))
        mlflow.log_metric("best_pr_auc", float(best_row["pr_auc"]))

        summary_json = Path(project_root) / "Data" / "experiments" / "anomaly_day6" / run_name / "mlflow_summary.json"
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(
                {
                    "tracking_uri": tracking_uri,
                    "run_id": run.info.run_id,
                    "experiment_name": experiment_name,
                    "best_model": best_model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifact_paths = [*artifact_paths, summary_json]

        for artifact_path in artifact_paths:
            path = Path(artifact_path)
            if path.exists():
                mlflow.log_artifact(str(path))

        return run.info.run_id


def log_sequence_forecasting_run(
    *,
    metrics: dict[str, float],
    predictions: pd.DataFrame,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    prophet_config: dict[str, Any] | None,
    artifact_paths: list[Path],
    project_root: Path,
    experiment_name: str = "week2-sequence-forecasting",
    run_name: str | None = None,
    tracking_uri: str | None = None,
    variant_name: str | None = None,
) -> str:
    mlflow = _load_mlflow()
    tracking_uri = tracking_uri or default_tracking_uri(project_root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    effective_run_name = run_name or variant_name or "sequence_forecasting"
    with mlflow.start_run(run_name=effective_run_name) as run:
        mlflow.log_param("variant_name", variant_name or effective_run_name)
        for key, value in model_config.items():
            mlflow.log_param(f"model_{key}", value)
        for key, value in training_config.items():
            mlflow.log_param(f"training_{key}", value)
        if prophet_config is not None:
            for key, value in prophet_config.items():
                mlflow.log_param(f"prophet_{key}", value)

        for key, value in metrics.items():
            if value is not None and not pd.isna(value):
                mlflow.log_metric(key, float(value))

        if {"rul", "prediction"}.issubset(predictions.columns):
            calibration_metrics = compute_regression_calibration_metrics(
                targets=predictions["rul"].to_numpy(),
                predictions=predictions["prediction"].to_numpy(),
            )
            for key, value in calibration_metrics.items():
                if value is not None and not pd.isna(value):
                    mlflow.log_metric(key, float(value))

        summary_json = Path(project_root) / "Data" / "experiments" / "sequence_tracking" / f"{effective_run_name}_mlflow.json"
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(
                {
                    "tracking_uri": tracking_uri,
                    "run_id": run.info.run_id,
                    "experiment_name": experiment_name,
                    "variant_name": variant_name or effective_run_name,
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        for artifact_path in [*artifact_paths, summary_json]:
            path = Path(artifact_path)
            if path.exists():
                mlflow.log_artifact(str(path))

        return run.info.run_id

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

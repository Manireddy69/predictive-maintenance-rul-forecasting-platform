from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .maintenance_scheduler import (
    MaintenanceCostMatrix,
    SchedulerArtifacts,
    SchedulerResources,
    build_scheduler_tasks_from_predictions,
    run_sensitivity_analysis,
    save_scheduler_artifacts,
    solve_maintenance_schedule,
)
from .sequence_attention_model import (
    OptunaSearchConfig,
    ProphetEnsembleConfig,
    SavedSequenceArtifacts,
    SequenceModelConfig,
    SequenceTrainingConfig,
    train_day9_sequence_model,
)


@dataclass(frozen=True)
class Week2CheckpointArtifacts:
    output_dir: Path
    scheduler_artifacts: SchedulerArtifacts
    checkpoint_summary_json: Path
    maintenance_candidates_csv: Path
    predictions_csv: Path
    forecast_run_dir: Path | None = None


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parent.parent


def default_week2_output_root() -> Path:
    return project_root_from_here() / "Data" / "experiments" / "week2_checkpoint"


def find_latest_sequence_predictions(
    sequence_output_root: Path | None = None,
) -> Path | None:
    sequence_output_root = sequence_output_root or (project_root_from_here() / "Data" / "experiments" / "day9_sequence_training")
    if not sequence_output_root.exists():
        return None
    candidates = sorted(sequence_output_root.glob("*/test_predictions.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def build_maintenance_candidates_frame(
    predictions: pd.DataFrame,
    cycles_per_day: float = 5.0,
    planning_horizon_days: int = 14,
    cost_matrix: MaintenanceCostMatrix | None = None,
    max_tasks: int | None = 20,
) -> pd.DataFrame:
    tasks = build_scheduler_tasks_from_predictions(
        predictions=predictions,
        cycles_per_day=cycles_per_day,
        planning_horizon_days=planning_horizon_days,
        cost_matrix=cost_matrix,
        max_tasks=max_tasks,
    )
    return pd.DataFrame([task.__dict__ for task in tasks]).sort_values(["preferred_day", "predicted_rul_days"]).reset_index(drop=True)


def run_week2_checkpoint(
    fd: str = "FD001",
    run_name: str = "",
    output_dir: str = "Data/experiments/week2_checkpoint",
    predictions_csv: str | None = None,
    train_new_model: bool = False,
    cycles_per_day: float = 5.0,
    max_candidate_assets: int = 20,
    model_config: SequenceModelConfig | None = None,
    training_config: SequenceTrainingConfig | None = None,
    search_config: OptunaSearchConfig | None = None,
    prophet_config: ProphetEnsembleConfig | None = None,
    scheduler_resources: SchedulerResources | None = None,
    cost_matrix: MaintenanceCostMatrix | None = None,
) -> Week2CheckpointArtifacts:
    root = project_root_from_here()
    model_config = model_config or SequenceModelConfig()
    training_config = training_config or SequenceTrainingConfig(target_mode="rul")
    search_config = search_config or OptunaSearchConfig(n_trials=0)
    prophet_config = prophet_config or ProphetEnsembleConfig(enabled=False)
    scheduler_resources = scheduler_resources or SchedulerResources()
    cost_matrix = cost_matrix or MaintenanceCostMatrix()

    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    resolved_run_name = run_name.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    forecast_artifacts: SavedSequenceArtifacts | None = None
    prediction_path: Path
    predictions: pd.DataFrame

    if train_new_model:
        _, predictions, forecast_artifacts = train_day9_sequence_model(
            fd=fd,
            model_config=model_config,
            training_config=training_config,
            search_config=search_config,
            prophet_config=prophet_config,
            run_name=f"{resolved_run_name}_forecast",
            output_dir=str(root / "Data" / "experiments" / "day9_sequence_training"),
        )
        prediction_path = forecast_artifacts.predictions_csv
    else:
        if predictions_csv is not None:
            prediction_path = Path(predictions_csv)
            if not prediction_path.is_absolute():
                prediction_path = root / prediction_path
        else:
            latest = find_latest_sequence_predictions()
            if latest is None:
                raise FileNotFoundError(
                    "No sequence prediction artifacts were found. Provide `predictions_csv` or set `train_new_model=True`."
                )
            prediction_path = latest
        predictions = pd.read_csv(prediction_path)

    if not train_new_model:
        predictions = pd.read_csv(prediction_path)

    candidates = build_maintenance_candidates_frame(
        predictions=predictions,
        cycles_per_day=cycles_per_day,
        planning_horizon_days=scheduler_resources.planning_horizon_days,
        cost_matrix=cost_matrix,
        max_tasks=max_candidate_assets,
    )
    tasks = build_scheduler_tasks_from_predictions(
        predictions=predictions,
        cycles_per_day=cycles_per_day,
        planning_horizon_days=scheduler_resources.planning_horizon_days,
        cost_matrix=cost_matrix,
        max_tasks=max_candidate_assets,
    )
    schedule_result = solve_maintenance_schedule(tasks=tasks, resources=scheduler_resources, cost_matrix=cost_matrix)
    sensitivity_frame = run_sensitivity_analysis(tasks=tasks, resources=scheduler_resources, base_cost_matrix=cost_matrix)
    scheduler_artifacts = save_scheduler_artifacts(
        output_root=output_root,
        run_name=resolved_run_name,
        schedule_result=schedule_result,
        sensitivity_frame=sensitivity_frame,
        tasks=tasks,
        resources=scheduler_resources,
        cost_matrix=cost_matrix,
    )

    maintenance_candidates_csv = run_dir / "maintenance_candidates.csv"
    candidates.to_csv(maintenance_candidates_csv, index=False)
    checkpoint_summary_json = run_dir / "week2_checkpoint_summary.json"
    checkpoint_summary_json.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "fd": fd,
                "prediction_source": str(prediction_path),
                "forecast_run_dir": str(forecast_artifacts.output_dir) if forecast_artifacts is not None else None,
                "scheduler_summary": schedule_result.summary,
                "sensitivity_scenarios": sensitivity_frame.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return Week2CheckpointArtifacts(
        output_dir=run_dir,
        scheduler_artifacts=scheduler_artifacts,
        checkpoint_summary_json=checkpoint_summary_json,
        maintenance_candidates_csv=maintenance_candidates_csv,
        predictions_csv=prediction_path,
        forecast_run_dir=forecast_artifacts.output_dir if forecast_artifacts is not None else None,
    )

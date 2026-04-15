from __future__ import annotations

import argparse
import json

from .maintenance_scheduler import MaintenanceCostMatrix, SchedulerResources
from .sequence_attention_model import OptunaSearchConfig, ProphetEnsembleConfig, SequenceModelConfig, SequenceTrainingConfig
from .week2_checkpoint import run_week2_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the week 2 checkpoint pipeline from predictions to maintenance schedule."
    )
    parser.add_argument("--fd", type=str, default="FD001", help="CMAPSS subset to use when training a fresh forecast model.")
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="",
        help="Optional path to an existing sequence prediction CSV. If omitted, the latest saved run is used.",
    )
    parser.add_argument(
        "--train-new-model",
        action="store_true",
        help="Train a fresh forecasting model before scheduling instead of using an existing predictions CSV.",
    )
    parser.add_argument("--run-name", type=str, default="", help="Optional output folder name.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Data/experiments/week2_checkpoint",
        help="Directory for week 2 checkpoint artifacts.",
    )
    parser.add_argument("--cycles-per-day", type=float, default=5.0, help="Approximate cycle-to-day conversion factor.")
    parser.add_argument(
        "--max-candidate-assets",
        type=int,
        default=20,
        help="Maximum number of highest-risk assets to send into the scheduler.",
    )
    parser.add_argument("--planning-horizon-days", type=int, default=14, help="Maintenance planning horizon in days.")
    parser.add_argument("--technician-hours-per-day", type=float, default=24.0, help="Total technician hours available per day.")
    parser.add_argument("--max-daily-downtime-hours", type=float, default=16.0, help="Production SLA for planned downtime per day.")
    parser.add_argument("--downtime-cost-per-hour", type=float, default=10_000.0, help="Planned and unplanned downtime cost per hour.")
    parser.add_argument("--technician-hourly-rate", type=float, default=150.0, help="Technician labor rate.")
    parser.add_argument("--max-epochs", type=int, default=5, help="Epoch cap when --train-new-model is enabled.")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Optuna trials when --train-new-model is enabled.")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size when training a fresh model.")
    parser.add_argument("--num-layers", type=int, default=2, choices=[2, 3], help="Number of bidirectional LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate when training a fresh model.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate when training a fresh model.")
    parser.add_argument(
        "--disable-prophet-ensemble",
        action="store_true",
        help="Disable Prophet blending when training a fresh model for the checkpoint.",
    )
    parser.add_argument(
        "--prophet-fit-timeout-seconds",
        type=int,
        default=60,
        help="Timeout for Prophet fitting when the checkpoint trains a fresh hybrid forecast model.",
    )
    parser.add_argument(
        "--strict-prophet",
        action="store_true",
        help="Fail the checkpoint if Prophet cannot finish instead of falling back to LSTM-only predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_config = SequenceModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    training_config = SequenceTrainingConfig(target_mode="rul", max_epochs=args.max_epochs)
    search_config = OptunaSearchConfig(n_trials=args.optuna_trials)
    prophet_config = ProphetEnsembleConfig(
        enabled=not args.disable_prophet_ensemble,
        fit_timeout_seconds=args.prophet_fit_timeout_seconds,
        fallback_to_lstm_on_failure=not args.strict_prophet,
    )
    scheduler_resources = SchedulerResources(
        planning_horizon_days=args.planning_horizon_days,
        technician_hours_per_day=args.technician_hours_per_day,
        max_daily_downtime_hours=args.max_daily_downtime_hours,
    )
    cost_matrix = MaintenanceCostMatrix(
        downtime_cost_per_hour=args.downtime_cost_per_hour,
        technician_hourly_rate=args.technician_hourly_rate,
    )

    artifacts = run_week2_checkpoint(
        fd=args.fd,
        run_name=args.run_name,
        output_dir=args.output_dir,
        predictions_csv=args.predictions_csv.strip() or None,
        train_new_model=args.train_new_model,
        cycles_per_day=args.cycles_per_day,
        max_candidate_assets=args.max_candidate_assets,
        model_config=model_config,
        training_config=training_config,
        search_config=search_config,
        prophet_config=prophet_config,
        scheduler_resources=scheduler_resources,
        cost_matrix=cost_matrix,
    )

    print("=== Week 2 checkpoint ===")
    print(f"Output dir: {artifacts.output_dir}")
    print(f"Predictions source: {artifacts.predictions_csv}")
    print(f"Maintenance candidates: {artifacts.maintenance_candidates_csv}")
    print(f"Schedule JSON: {artifacts.scheduler_artifacts.schedule_json}")
    print(f"Sensitivity CSV: {artifacts.scheduler_artifacts.sensitivity_csv}")
    print(f"Trade-off chart: {artifacts.scheduler_artifacts.tradeoff_png}")
    if artifacts.forecast_run_dir is not None:
        print(f"Forecast run dir: {artifacts.forecast_run_dir}")
    print(json.dumps(json.loads(artifacts.checkpoint_summary_json.read_text(encoding='utf-8'))["scheduler_summary"], indent=2))


if __name__ == "__main__":
    main()

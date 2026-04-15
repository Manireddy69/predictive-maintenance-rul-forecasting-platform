from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _latest_file(pattern: str) -> Path | None:
    matches = sorted(project_root().glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def latest_week2_schedule_json() -> Path | None:
    return _latest_file("Data/experiments/week2_checkpoint/*/optimal_schedule.json")


def latest_week2_schedule_csv() -> Path | None:
    return _latest_file("Data/experiments/week2_checkpoint/*/optimal_schedule.csv")


def latest_week2_sensitivity_csv() -> Path | None:
    return _latest_file("Data/experiments/week2_checkpoint/*/sensitivity_analysis.csv")


def latest_sequence_predictions_csv() -> Path | None:
    return _latest_file("Data/experiments/day9_sequence_training/*/test_predictions.csv")


def load_schedule_summary() -> dict[str, object]:
    schedule_json = latest_week2_schedule_json()
    if schedule_json is None:
        return {
            "task_count": 0,
            "total_cost": 0.0,
            "total_risk_cost": 0.0,
            "on_or_before_preferred_rate": 0.0,
            "solver_status": "No schedule artifacts found",
        }
    payload = json.loads(schedule_json.read_text(encoding="utf-8"))
    return payload.get("summary", {})


def load_schedule_table() -> pd.DataFrame:
    schedule_csv = latest_week2_schedule_csv()
    if schedule_csv is None:
        return pd.DataFrame(
            {
                "equipment_id": ["unit_1"],
                "scheduled_day": [0],
                "repair_type": ["inspection"],
                "predicted_rul_days": [7.0],
                "total_cost": [12_500.0],
            }
        )
    return pd.read_csv(schedule_csv)


def load_sensitivity_table() -> pd.DataFrame:
    sensitivity_csv = latest_week2_sensitivity_csv()
    if sensitivity_csv is None:
        return pd.DataFrame(
            {
                "scenario": ["base", "high_downtime", "high_labor"],
                "total_cost": [120_000.0, 145_000.0, 130_000.0],
                "total_risk_cost": [15_000.0, 9_000.0, 14_000.0],
                "schedule_robustness": [1.0, 0.75, 0.9],
            }
        )
    return pd.read_csv(sensitivity_csv)


def load_prediction_table() -> pd.DataFrame:
    predictions_csv = latest_sequence_predictions_csv()
    if predictions_csv is None:
        return pd.DataFrame(
            {
                "unit": [1, 2, 3],
                "target_cycle": [50, 50, 50],
                "prediction": [24.0, 12.0, 5.0],
                "rul": [22.0, 15.0, 6.0],
            }
        )
    return pd.read_csv(predictions_csv)

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.maintenance_scheduler import (
    MaintenanceCostMatrix,
    SchedulerResources,
    build_scheduler_tasks_from_predictions,
    run_sensitivity_analysis,
    save_scheduler_artifacts,
    solve_maintenance_schedule,
)


def _make_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": [1, 1, 2, 2, 3, 3],
            "target_cycle": [10, 12, 10, 12, 10, 12],
            "prediction": [60.0, 18.0, 55.0, 8.0, 45.0, 3.0],
            "rul": [58.0, 16.0, 54.0, 10.0, 43.0, 4.0],
        }
    )


def test_build_scheduler_tasks_from_predictions_uses_latest_window_per_unit() -> None:
    tasks = build_scheduler_tasks_from_predictions(_make_prediction_frame(), cycles_per_day=5.0, planning_horizon_days=14)

    assert len(tasks) == 3
    assert {task.equipment_id for task in tasks} == {"unit_1", "unit_2", "unit_3"}
    assert tasks[0].repair_type in {"minor_service", "inspection", "component_repair", "major_overhaul"}


def test_solve_maintenance_schedule_respects_daily_resource_constraints() -> None:
    tasks = build_scheduler_tasks_from_predictions(_make_prediction_frame(), cycles_per_day=5.0, planning_horizon_days=5)
    resources = SchedulerResources(
        planning_horizon_days=5,
        technician_hours_per_day=16.0,
        max_daily_downtime_hours=8.0,
    )
    result = solve_maintenance_schedule(tasks=tasks, resources=resources, cost_matrix=MaintenanceCostMatrix())

    daily_downtime = result.schedule.groupby("scheduled_day")["duration_hours"].sum()
    daily_tech_hours = (result.schedule["duration_hours"] * result.schedule["required_technicians"]).groupby(result.schedule["scheduled_day"]).sum()
    assert (daily_downtime <= 8.0 + 1e-9).all()
    assert (daily_tech_hours <= 16.0 + 1e-9).all()
    assert result.summary["task_count"] == 3


def test_run_sensitivity_analysis_reports_schedule_robustness() -> None:
    tasks = build_scheduler_tasks_from_predictions(_make_prediction_frame(), cycles_per_day=5.0, planning_horizon_days=5)
    resources = SchedulerResources(planning_horizon_days=5, technician_hours_per_day=16.0, max_daily_downtime_hours=8.0)

    sensitivity = run_sensitivity_analysis(tasks=tasks, resources=resources, base_cost_matrix=MaintenanceCostMatrix())

    assert {"scenario", "total_cost", "total_risk_cost", "schedule_robustness"}.issubset(sensitivity.columns)
    assert "base" in sensitivity["scenario"].tolist()


def test_save_scheduler_artifacts_writes_json_csv_and_plot() -> None:
    tasks = build_scheduler_tasks_from_predictions(_make_prediction_frame(), cycles_per_day=5.0, planning_horizon_days=5)
    resources = SchedulerResources(planning_horizon_days=5, technician_hours_per_day=16.0, max_daily_downtime_hours=8.0)
    cost_matrix = MaintenanceCostMatrix()
    result = solve_maintenance_schedule(tasks=tasks, resources=resources, cost_matrix=cost_matrix)
    sensitivity = run_sensitivity_analysis(tasks=tasks, resources=resources, base_cost_matrix=cost_matrix)
    output_root = Path("Data/test_artifacts/scheduler_tests")
    if output_root.exists():
        shutil.rmtree(output_root)

    artifacts = save_scheduler_artifacts(
        output_root=output_root,
        run_name="scheduler_test",
        schedule_result=result,
        sensitivity_frame=sensitivity,
        tasks=tasks,
        resources=resources,
        cost_matrix=cost_matrix,
    )

    assert artifacts.schedule_json.exists()
    assert artifacts.schedule_csv.exists()
    assert artifacts.sensitivity_json.exists()
    assert artifacts.sensitivity_csv.exists()
    assert artifacts.tradeoff_png.exists()
    assert artifacts.summary_markdown.exists()

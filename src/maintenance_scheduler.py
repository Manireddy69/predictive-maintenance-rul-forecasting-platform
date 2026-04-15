from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_pulp_runtime() -> Any:
    try:
        import pulp
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "PuLP is not installed. Add `pulp` to the environment before running the maintenance scheduler."
        ) from exc
    return pulp


@dataclass(frozen=True)
class RepairTypeProfile:
    repair_type: str
    repair_cost: float
    duration_hours: float
    required_technicians: int
    part_name: str
    part_quantity: int = 1


def default_repair_profiles() -> dict[str, RepairTypeProfile]:
    return {
        "inspection": RepairTypeProfile(
            repair_type="inspection",
            repair_cost=2_000.0,
            duration_hours=2.0,
            required_technicians=1,
            part_name="inspection_kit",
        ),
        "minor_service": RepairTypeProfile(
            repair_type="minor_service",
            repair_cost=5_000.0,
            duration_hours=4.0,
            required_technicians=1,
            part_name="service_kit",
        ),
        "component_repair": RepairTypeProfile(
            repair_type="component_repair",
            repair_cost=9_000.0,
            duration_hours=6.0,
            required_technicians=2,
            part_name="rotor_assembly",
        ),
        "major_overhaul": RepairTypeProfile(
            repair_type="major_overhaul",
            repair_cost=15_000.0,
            duration_hours=8.0,
            required_technicians=2,
            part_name="engine_core",
        ),
    }


def default_daily_production_multipliers(planning_horizon_days: int) -> tuple[float, ...]:
    weekly_pattern = (1.15, 1.10, 1.05, 1.00, 1.00, 0.85, 0.80)
    return tuple(weekly_pattern[day % len(weekly_pattern)] for day in range(planning_horizon_days))


@dataclass(frozen=True)
class MaintenanceCostMatrix:
    downtime_cost_per_hour: float = 10_000.0
    technician_hourly_rate: float = 150.0
    repair_profiles: dict[str, RepairTypeProfile] = field(default_factory=default_repair_profiles)

    def __post_init__(self) -> None:
        if self.downtime_cost_per_hour <= 0:
            raise ValueError("downtime_cost_per_hour must be positive.")
        if self.technician_hourly_rate <= 0:
            raise ValueError("technician_hourly_rate must be positive.")
        if not self.repair_profiles:
            raise ValueError("repair_profiles must include at least one repair type.")


@dataclass(frozen=True)
class SchedulerResources:
    planning_horizon_days: int = 14
    technician_hours_per_day: float = 24.0
    max_daily_downtime_hours: float = 16.0
    parts_inventory: dict[str, int] = field(
        default_factory=lambda: {
            "inspection_kit": 50,
            "service_kit": 50,
            "rotor_assembly": 50,
            "engine_core": 50,
        }
    )
    daily_production_multipliers: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.planning_horizon_days < 1:
            raise ValueError("planning_horizon_days must be positive.")
        if self.technician_hours_per_day <= 0:
            raise ValueError("technician_hours_per_day must be positive.")
        if self.max_daily_downtime_hours <= 0:
            raise ValueError("max_daily_downtime_hours must be positive.")
        if self.daily_production_multipliers and len(self.daily_production_multipliers) != self.planning_horizon_days:
            raise ValueError("daily_production_multipliers must match planning_horizon_days.")

    def production_multiplier_for_day(self, day: int) -> float:
        if not self.daily_production_multipliers:
            return default_daily_production_multipliers(self.planning_horizon_days)[day]
        return float(self.daily_production_multipliers[day])


@dataclass(frozen=True)
class MaintenanceTask:
    equipment_id: str
    repair_type: str
    predicted_rul: float
    predicted_rul_days: float
    preferred_day: int
    risk_penalty_per_day: float
    criticality: float

    def __post_init__(self) -> None:
        if not self.equipment_id:
            raise ValueError("equipment_id must be provided.")
        if self.predicted_rul < 0:
            raise ValueError("predicted_rul must be non-negative.")
        if self.predicted_rul_days < 0:
            raise ValueError("predicted_rul_days must be non-negative.")
        if self.preferred_day < 0:
            raise ValueError("preferred_day must be non-negative.")
        if self.risk_penalty_per_day < 0:
            raise ValueError("risk_penalty_per_day must be non-negative.")
        if not 0.0 <= self.criticality <= 1.5:
            raise ValueError("criticality must stay in a reasonable range.")


@dataclass(frozen=True)
class SchedulerArtifacts:
    output_dir: Path
    schedule_json: Path
    schedule_csv: Path
    sensitivity_json: Path
    sensitivity_csv: Path
    tradeoff_png: Path
    summary_markdown: Path


@dataclass(frozen=True)
class MaintenanceScheduleResult:
    schedule: pd.DataFrame
    summary: dict[str, float | int | str]
    solver_status: str


def _select_prediction_column(predictions: pd.DataFrame) -> str:
    for column in ("hybrid_prediction", "prediction", "lstm_prediction"):
        if column in predictions.columns:
            return column
    raise ValueError("Could not find a RUL prediction column in the predictions dataframe.")


def _select_latest_prediction_per_unit(predictions: pd.DataFrame) -> pd.DataFrame:
    if "unit" not in predictions.columns:
        raise ValueError("Predictions must include a `unit` column to build maintenance tasks.")

    working = predictions.copy()
    sort_columns = [column for column in ("unit", "target_cycle", "window_end_index") if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns)
    return working.groupby("unit", as_index=False).tail(1).reset_index(drop=True)


def infer_repair_type(predicted_rul_days: float) -> str:
    if predicted_rul_days <= 3:
        return "major_overhaul"
    if predicted_rul_days <= 7:
        return "component_repair"
    if predicted_rul_days <= 14:
        return "minor_service"
    return "inspection"


def build_scheduler_tasks_from_predictions(
    predictions: pd.DataFrame,
    cycles_per_day: float = 5.0,
    planning_horizon_days: int = 14,
    cost_matrix: MaintenanceCostMatrix | None = None,
    max_tasks: int | None = 20,
) -> list[MaintenanceTask]:
    if cycles_per_day <= 0:
        raise ValueError("cycles_per_day must be positive.")

    cost_matrix = cost_matrix or MaintenanceCostMatrix()
    latest_predictions = _select_latest_prediction_per_unit(predictions)
    prediction_column = _select_prediction_column(latest_predictions)
    latest_predictions = latest_predictions.sort_values(prediction_column, ascending=True).reset_index(drop=True)
    if max_tasks is not None:
        latest_predictions = latest_predictions.head(max_tasks).reset_index(drop=True)

    tasks: list[MaintenanceTask] = []
    for row in latest_predictions.itertuples(index=False):
        predicted_rul = max(float(getattr(row, prediction_column)), 0.0)
        predicted_rul_days = max(predicted_rul / cycles_per_day, 0.5)
        repair_type = infer_repair_type(predicted_rul_days)
        criticality = float(np.clip((14.0 - predicted_rul_days) / 14.0, 0.1, 1.0))
        preferred_day = int(np.clip(math.floor(max(predicted_rul_days - 1.0, 0.0)), 0, planning_horizon_days - 1))
        risk_penalty_per_day = cost_matrix.downtime_cost_per_hour * (0.2 + criticality)
        tasks.append(
            MaintenanceTask(
                equipment_id=f"unit_{int(getattr(row, 'unit'))}",
                repair_type=repair_type,
                predicted_rul=predicted_rul,
                predicted_rul_days=predicted_rul_days,
                preferred_day=preferred_day,
                risk_penalty_per_day=risk_penalty_per_day,
                criticality=criticality,
            )
        )
    return tasks


def _task_day_cost(
    task: MaintenanceTask,
    day: int,
    resources: SchedulerResources,
    cost_matrix: MaintenanceCostMatrix,
) -> tuple[float, float, float, float]:
    profile = cost_matrix.repair_profiles[task.repair_type]
    production_multiplier = resources.production_multiplier_for_day(day)
    repair_cost = profile.repair_cost
    labor_cost = profile.duration_hours * profile.required_technicians * cost_matrix.technician_hourly_rate
    planned_downtime_cost = profile.duration_hours * cost_matrix.downtime_cost_per_hour * production_multiplier
    risk_cost = max(day - task.preferred_day, 0) * task.risk_penalty_per_day
    total_cost = repair_cost + labor_cost + planned_downtime_cost + risk_cost
    return total_cost, repair_cost + labor_cost + planned_downtime_cost, risk_cost, production_multiplier


def solve_maintenance_schedule(
    tasks: list[MaintenanceTask],
    resources: SchedulerResources | None = None,
    cost_matrix: MaintenanceCostMatrix | None = None,
) -> MaintenanceScheduleResult:
    if not tasks:
        raise ValueError("At least one maintenance task is required to build a schedule.")

    resources = resources or SchedulerResources()
    cost_matrix = cost_matrix or MaintenanceCostMatrix()
    pulp = _load_pulp_runtime()

    model = pulp.LpProblem("maintenance_schedule", pulp.LpMinimize)
    task_lookup = {task.equipment_id: task for task in tasks}
    task_days = range(resources.planning_horizon_days)
    decision = {
        (task.equipment_id, day): pulp.LpVariable(f"x_{task.equipment_id}_{day}", lowBound=0, upBound=1, cat="Binary")
        for task in tasks
        for day in task_days
    }

    model += pulp.lpSum(
        _task_day_cost(task, day, resources, cost_matrix)[0] * decision[(task.equipment_id, day)]
        for task in tasks
        for day in task_days
    )

    for task in tasks:
        model += (
            pulp.lpSum(decision[(task.equipment_id, day)] for day in task_days) == 1,
            f"schedule_once_{task.equipment_id}",
        )

    for day in task_days:
        model += (
            pulp.lpSum(
                cost_matrix.repair_profiles[task.repair_type].duration_hours
                * cost_matrix.repair_profiles[task.repair_type].required_technicians
                * decision[(task.equipment_id, day)]
                for task in tasks
            )
            <= resources.technician_hours_per_day,
            f"technician_hours_day_{day}",
        )
        model += (
            pulp.lpSum(
                cost_matrix.repair_profiles[task.repair_type].duration_hours * decision[(task.equipment_id, day)]
                for task in tasks
            )
            <= resources.max_daily_downtime_hours,
            f"downtime_sla_day_{day}",
        )

    parts_to_track = sorted(
        {
            cost_matrix.repair_profiles[task.repair_type].part_name
            for task in tasks
            if cost_matrix.repair_profiles[task.repair_type].part_name in resources.parts_inventory
        }
    )
    for part_name in parts_to_track:
        model += (
            pulp.lpSum(
                cost_matrix.repair_profiles[task.repair_type].part_quantity
                * decision[(task.equipment_id, day)]
                for task in tasks
                for day in task_days
                if cost_matrix.repair_profiles[task.repair_type].part_name == part_name
            )
            <= resources.parts_inventory[part_name],
            f"inventory_{part_name}",
        )

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    solver_status = pulp.LpStatus[model.status]
    if solver_status != "Optimal":
        raise RuntimeError(f"Unable to find an optimal maintenance schedule. Solver status: {solver_status}")

    schedule_rows: list[dict[str, float | int | str]] = []
    total_cost = 0.0
    total_risk_cost = 0.0
    total_direct_cost = 0.0
    total_downtime_hours = 0.0
    total_technician_hours = 0.0

    for equipment_id, task in task_lookup.items():
        assigned_day = next(day for day in task_days if decision[(equipment_id, day)].value() and decision[(equipment_id, day)].value() > 0.5)
        profile = cost_matrix.repair_profiles[task.repair_type]
        schedule_cost, direct_cost, risk_cost, production_multiplier = _task_day_cost(task, assigned_day, resources, cost_matrix)
        total_cost += schedule_cost
        total_direct_cost += direct_cost
        total_risk_cost += risk_cost
        total_downtime_hours += profile.duration_hours
        total_technician_hours += profile.duration_hours * profile.required_technicians
        schedule_rows.append(
            {
                "equipment_id": equipment_id,
                "scheduled_day": assigned_day,
                "repair_type": task.repair_type,
                "predicted_rul": task.predicted_rul,
                "predicted_rul_days": task.predicted_rul_days,
                "preferred_day": task.preferred_day,
                "criticality": task.criticality,
                "production_multiplier": production_multiplier,
                "direct_cost": round(direct_cost, 2),
                "risk_cost": round(risk_cost, 2),
                "total_cost": round(schedule_cost, 2),
                "duration_hours": profile.duration_hours,
                "required_technicians": profile.required_technicians,
                "part_name": profile.part_name,
                "part_quantity": profile.part_quantity,
            }
        )

    schedule = pd.DataFrame(schedule_rows).sort_values(["scheduled_day", "criticality"], ascending=[True, False]).reset_index(drop=True)
    changed_after_preferred = int((schedule["scheduled_day"] > schedule["preferred_day"]).sum())
    summary: dict[str, float | int | str] = {
        "solver_status": solver_status,
        "task_count": int(len(schedule)),
        "total_cost": round(total_cost, 2),
        "total_direct_cost": round(total_direct_cost, 2),
        "total_risk_cost": round(total_risk_cost, 2),
        "average_cost_per_task": round(total_cost / len(schedule), 2),
        "tasks_after_preferred_day": changed_after_preferred,
        "on_or_before_preferred_rate": round(float((schedule["scheduled_day"] <= schedule["preferred_day"]).mean()), 4),
        "total_downtime_hours": round(total_downtime_hours, 2),
        "total_technician_hours": round(total_technician_hours, 2),
    }
    return MaintenanceScheduleResult(schedule=schedule, summary=summary, solver_status=solver_status)


def run_sensitivity_analysis(
    tasks: list[MaintenanceTask],
    resources: SchedulerResources,
    base_cost_matrix: MaintenanceCostMatrix,
    scenario_overrides: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    base_schedule = solve_maintenance_schedule(tasks=tasks, resources=resources, cost_matrix=base_cost_matrix)
    base_assignment = dict(zip(base_schedule.schedule["equipment_id"], base_schedule.schedule["scheduled_day"], strict=False))

    scenario_overrides = scenario_overrides or {
        "base": {"downtime_multiplier": 1.0, "repair_multiplier": 1.0, "labor_multiplier": 1.0},
        "high_downtime": {"downtime_multiplier": 1.25, "repair_multiplier": 1.0, "labor_multiplier": 1.0},
        "low_downtime": {"downtime_multiplier": 0.85, "repair_multiplier": 1.0, "labor_multiplier": 1.0},
        "high_parts": {"downtime_multiplier": 1.0, "repair_multiplier": 1.20, "labor_multiplier": 1.0},
        "high_labor": {"downtime_multiplier": 1.0, "repair_multiplier": 1.0, "labor_multiplier": 1.20},
        "stress_case": {"downtime_multiplier": 1.30, "repair_multiplier": 1.15, "labor_multiplier": 1.15},
    }

    rows: list[dict[str, float | int | str]] = []
    for scenario_name, override in scenario_overrides.items():
        repair_multiplier = float(override.get("repair_multiplier", 1.0))
        scenario_profiles = {
            name: RepairTypeProfile(
                repair_type=profile.repair_type,
                repair_cost=profile.repair_cost * repair_multiplier,
                duration_hours=profile.duration_hours,
                required_technicians=profile.required_technicians,
                part_name=profile.part_name,
                part_quantity=profile.part_quantity,
            )
            for name, profile in base_cost_matrix.repair_profiles.items()
        }
        scenario_cost_matrix = MaintenanceCostMatrix(
            downtime_cost_per_hour=base_cost_matrix.downtime_cost_per_hour * float(override.get("downtime_multiplier", 1.0)),
            technician_hourly_rate=base_cost_matrix.technician_hourly_rate * float(override.get("labor_multiplier", 1.0)),
            repair_profiles=scenario_profiles,
        )
        result = solve_maintenance_schedule(tasks=tasks, resources=resources, cost_matrix=scenario_cost_matrix)
        current_assignment = result.schedule.set_index("equipment_id")["scheduled_day"]
        base_assignment_series = pd.Series(base_assignment, name="scheduled_day").sort_index()
        changed_assignments = int(current_assignment.reindex(base_assignment_series.index).ne(base_assignment_series).sum())
        rows.append(
            {
                "scenario": scenario_name,
                "total_cost": result.summary["total_cost"],
                "total_risk_cost": result.summary["total_risk_cost"],
                "tasks_after_preferred_day": result.summary["tasks_after_preferred_day"],
                "on_or_before_preferred_rate": result.summary["on_or_before_preferred_rate"],
                "changed_assignments": changed_assignments,
                "schedule_robustness": round(1.0 - (changed_assignments / max(len(tasks), 1)), 4),
                "downtime_multiplier": override.get("downtime_multiplier", 1.0),
                "repair_multiplier": override.get("repair_multiplier", 1.0),
                "labor_multiplier": override.get("labor_multiplier", 1.0),
            }
        )

    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


def plot_cost_vs_risk_tradeoff(
    sensitivity_frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(sensitivity_frame["total_risk_cost"], sensitivity_frame["total_cost"], color="#005f73", s=80)
    for row in sensitivity_frame.itertuples(index=False):
        ax.annotate(str(row.scenario), (row.total_risk_cost, row.total_cost), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Total Risk Cost")
    ax.set_ylabel("Total Schedule Cost")
    ax.set_title("Maintenance Schedule Cost vs Risk Trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_scheduler_artifacts(
    output_root: Path,
    run_name: str,
    schedule_result: MaintenanceScheduleResult,
    sensitivity_frame: pd.DataFrame,
    tasks: list[MaintenanceTask],
    resources: SchedulerResources,
    cost_matrix: MaintenanceCostMatrix,
) -> SchedulerArtifacts:
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_json = output_dir / "optimal_schedule.json"
    schedule_csv = output_dir / "optimal_schedule.csv"
    sensitivity_json = output_dir / "sensitivity_analysis.json"
    sensitivity_csv = output_dir / "sensitivity_analysis.csv"
    tradeoff_png = output_dir / "cost_vs_risk_tradeoff.png"
    summary_markdown = output_dir / "scheduler_summary.md"

    schedule_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": schedule_result.summary,
        "solver_status": schedule_result.solver_status,
        "cost_matrix": {
            "downtime_cost_per_hour": cost_matrix.downtime_cost_per_hour,
            "technician_hourly_rate": cost_matrix.technician_hourly_rate,
            "repair_profiles": {name: asdict(profile) for name, profile in cost_matrix.repair_profiles.items()},
        },
        "resources": asdict(resources),
        "tasks": [asdict(task) for task in tasks],
        "schedule": schedule_result.schedule.to_dict(orient="records"),
    }
    schedule_json.write_text(json.dumps(schedule_payload, indent=2), encoding="utf-8")
    schedule_result.schedule.to_csv(schedule_csv, index=False)
    sensitivity_json.write_text(json.dumps(sensitivity_frame.to_dict(orient="records"), indent=2), encoding="utf-8")
    sensitivity_frame.to_csv(sensitivity_csv, index=False)
    plot_cost_vs_risk_tradeoff(sensitivity_frame=sensitivity_frame, output_path=tradeoff_png)

    summary_lines = [
        "# Week 2 Maintenance Scheduler Summary",
        "",
        f"- Tasks scheduled: `{schedule_result.summary['task_count']}`",
        f"- Total schedule cost: `${float(schedule_result.summary['total_cost']):,.2f}`",
        f"- Total risk cost: `${float(schedule_result.summary['total_risk_cost']):,.2f}`",
        f"- On-time rate: `{float(schedule_result.summary['on_or_before_preferred_rate']):.2%}`",
        f"- Downtime cost assumption: `${cost_matrix.downtime_cost_per_hour:,.0f}/hour`",
        f"- Technician hourly rate: `${cost_matrix.technician_hourly_rate:,.0f}/hour`",
        "",
        "## Sensitivity Highlights",
        "",
    ]
    best_row = sensitivity_frame.sort_values(["total_cost", "total_risk_cost"]).iloc[0]
    robust_row = sensitivity_frame.sort_values(["schedule_robustness", "total_cost"], ascending=[False, True]).iloc[0]
    summary_lines.extend(
        [
            f"- Lowest-cost scenario: `{best_row['scenario']}` with `${float(best_row['total_cost']):,.2f}` total cost",
            f"- Most robust scenario: `{robust_row['scenario']}` with `{float(robust_row['schedule_robustness']):.2%}` schedule stability",
        ]
    )
    summary_markdown.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return SchedulerArtifacts(
        output_dir=output_dir,
        schedule_json=schedule_json,
        schedule_csv=schedule_csv,
        sensitivity_json=sensitivity_json,
        sensitivity_csv=sensitivity_csv,
        tradeoff_png=tradeoff_png,
        summary_markdown=summary_markdown,
    )

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def output_dir() -> Path:
    path = project_root() / "reports" / "screenshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_text_card(lines: list[str], output_path: Path, title: str, figsize: tuple[float, float] = (12, 7)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.text(0.02, 0.95, title, fontsize=22, fontweight="bold", color="#173a63", va="top")
    ax.text(0.02, 0.86, "\n".join(lines), fontsize=13, color="#17202a", va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def repo_structure_visual() -> None:
    lines = [
        "Project-1/",
        "|-- src/                  reusable pipelines, models, evaluation CLIs",
        "|-- app/                  Streamlit dashboard pages",
        "|-- Data/CMaps/           NASA CMAPSS raw data",
        "|-- Data/experiments/     saved anomaly, RUL, and scheduler artifacts",
        "|-- dags/                 Airflow batch ingestion DAG",
        "|-- db/init/              TimescaleDB schema scripts",
        "|-- notebooks/            EDA and model review notebooks",
        "|-- reports/              final report, compliance, evidence metrics",
        "|-- tests/                focused unit/integration tests",
        "|-- deploy/               Docker and runtime deployment helpers",
    ]
    save_text_card(lines, output_dir() / "01_repo_structure.png", "Repository Structure")


def eda_summary_visual() -> None:
    lines = [
        "Dataset: NASA CMAPSS FD001",
        "",
        "Train units: 100",
        "Test units: 100",
        "Train rows: 20,631",
        "Test rows: 13,096",
        "Train cycle range: 128 to 362",
        "Test cycle range: 31 to 303",
        "",
        "Constant columns:",
        "setting_3, sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19",
        "",
        "Strong degradation sensors:",
        "sensor_9, sensor_14, sensor_4, sensor_3",
    ]
    save_text_card(lines, output_dir() / "02_eda_summary.png", "EDA Summary")


def anomaly_results_visual() -> None:
    metrics = pd.read_csv(project_root() / "reports" / "anomaly_acceptance_metrics.csv")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("white")
    metrics = metrics.sort_values("f1", ascending=True)
    ax.barh(metrics["model_key"], metrics["f1"], color="#2a9d8f")
    ax.axvline(0.88, color="#c0392b", linestyle="--", linewidth=2, label="F1 target 0.88")
    ax.set_xlim(0.65, 1.0)
    ax.set_xlabel("F1 Score")
    ax.set_title("Anomaly Detection Acceptance Metrics", fontsize=18, fontweight="bold", color="#173a63")
    for index, value in enumerate(metrics["f1"]):
        ax.text(value + 0.005, index, f"{value:.3f}", va="center", fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir() / "03_anomaly_results.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def lstm_autoencoder_visual() -> None:
    metrics = pd.read_csv(project_root() / "reports" / "anomaly_acceptance_metrics.csv")
    row = metrics[metrics["model_key"] == "lstm_autoencoder"].iloc[0]
    lines = [
        "LSTM Autoencoder window-level anomaly result",
        "",
        f"ROC-AUC:          {row['roc_auc']:.4f}",
        f"PR-AUC:           {row['pr_auc']:.4f}",
        f"F1:               {row['f1']:.4f}",
        f"Precision:        {row['precision']:.4f}",
        f"Recall:           {row['recall']:.4f}",
        f"False alarm rate: {row['false_alarm_rate']:.4f}",
        "",
        "Interpretation:",
        "Best ranking metrics and lowest false alarm rate,",
        "while MAD/z-score tie the best threshold F1.",
    ]
    save_text_card(lines, output_dir() / "04_lstm_autoencoder.png", "LSTM Autoencoder Evidence")


def rul_metrics_visual() -> None:
    metrics = json.loads((project_root() / "reports" / "rul_acceptance_metrics.json").read_text(encoding="utf-8"))
    lines = [
        "Latest RUL forecasting metrics",
        "",
        f"Rows evaluated:              {metrics['row_count']:,}",
        f"RMSE:                        {metrics['rmse']:.2f}",
        f"MAE:                         {metrics['mae']:.2f}",
        f"R2:                          {metrics['r2']:.3f}",
        f"MAPE:                        {metrics['mape_percent']:.2f}%",
        f"Median absolute error:       {metrics['median_absolute_error']:.2f} cycles",
        f"Within 20 cycles:            {metrics['within_20_cycles_rate']:.2%}",
        "",
        "Status vs PDF target:",
        "MAPE target <= 12% is not met yet.",
    ]
    save_text_card(lines, output_dir() / "05_rul_metrics.png", "RUL Forecasting Metrics")


def scheduler_summary_visual() -> None:
    summary = json.loads(
        (project_root() / "Data" / "experiments" / "week2_checkpoint" / "week2_smoke" / "week2_checkpoint_summary.json").read_text(
            encoding="utf-8"
        )
    )["scheduler_summary"]
    lines = [
        "Week 2 maintenance scheduler result",
        "",
        f"Solver status:                 {summary['solver_status']}",
        f"Scheduled tasks:               {summary['task_count']}",
        f"Total cost:                    ${summary['total_cost']:,.2f}",
        f"Direct cost:                   ${summary['total_direct_cost']:,.2f}",
        f"Risk cost:                     ${summary['total_risk_cost']:,.2f}",
        f"Average cost per task:         ${summary['average_cost_per_task']:,.2f}",
        f"Tasks after preferred day:     {summary['tasks_after_preferred_day']}",
        f"On/before preferred rate:      {summary['on_or_before_preferred_rate']:.1%}",
        f"Total downtime hours:          {summary['total_downtime_hours']}",
        f"Total technician hours:        {summary['total_technician_hours']}",
    ]
    save_text_card(lines, output_dir() / "06_scheduler_summary.png", "Maintenance Scheduler Summary")


def dashboard_home_visual() -> None:
    summary = json.loads(
        (project_root() / "Data" / "experiments" / "week2_checkpoint" / "week2_smoke" / "week2_checkpoint_summary.json").read_text(
            encoding="utf-8"
        )
    )["scheduler_summary"]
    lines = [
        "Predictive Maintenance Dashboard",
        "",
        f"Scheduled tasks:     {summary['task_count']}",
        f"Total cost:          ${summary['total_cost']:,.0f}",
        f"Risk cost:           ${summary['total_risk_cost']:,.0f}",
        f"On-time rate:        {summary['on_or_before_preferred_rate']:.1%}",
        "",
        "Dashboard pages:",
        "- Overview: schedule health and sensitivity analysis",
        "- Equipment Detail: unit-level forecast and recommendation",
        "- Alerts Configuration: threshold and escalation draft",
        "- Reports: exportable schedule and sensitivity artifacts",
    ]
    save_text_card(lines, output_dir() / "08_dashboard_home.png", "Dashboard Home")


def dashboard_alerts_visual() -> None:
    lines = [
        "Alerts Configuration",
        "",
        "Configured draft controls:",
        "- Predicted RUL alert threshold: 7 days",
        "- Risk cost escalation threshold: $25,000",
        "- Daily downtime SLA threshold: 16 hours",
        "- Notification email: maintenance@example.com",
        "",
        "Current status:",
        "The UI captures alert rules for demo purposes.",
        "Persistence and real email/SMS/Slack delivery are documented as next iteration work.",
    ]
    save_text_card(lines, output_dir() / "11_dashboard_alerts.png", "Dashboard Alerts Page")


def dashboard_reports_visual() -> None:
    schedule_json = project_root() / "Data" / "experiments" / "week2_checkpoint" / "week2_smoke" / "optimal_schedule.json"
    sensitivity_csv = project_root() / "Data" / "experiments" / "week2_checkpoint" / "week2_smoke" / "sensitivity_analysis.csv"
    lines = [
        "Reports Page",
        "",
        "Exportable artifacts available:",
        f"- Schedule JSON: {schedule_json.name}",
        f"- Sensitivity CSV: {sensitivity_csv.name}",
        "- Scheduler summary markdown",
        "- Cost vs risk trade-off plot",
        "",
        "Report purpose:",
        "This page gives reviewers direct access to the saved schedule and sensitivity outputs",
        "that support the final maintenance optimization story.",
    ]
    save_text_card(lines, output_dir() / "12_dashboard_reports.png", "Dashboard Reports Page")


def copy_tradeoff_plot() -> None:
    source = project_root() / "Data" / "experiments" / "week2_checkpoint" / "week2_smoke" / "cost_vs_risk_tradeoff.png"
    destination = output_dir() / "07_cost_risk_plot.png"
    if source.exists():
        shutil.copyfile(source, destination)


def main() -> None:
    repo_structure_visual()
    eda_summary_visual()
    anomaly_results_visual()
    lstm_autoencoder_visual()
    rul_metrics_visual()
    scheduler_summary_visual()
    copy_tradeoff_plot()
    dashboard_home_visual()
    dashboard_alerts_visual()
    dashboard_reports_visual()
    print(f"Submission visuals written to: {output_dir()}")


if __name__ == "__main__":
    main()

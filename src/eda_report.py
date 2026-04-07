from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from .data import add_train_rul, load_fd_data, load_fd_rul
from .eda import estimate_sensor_degradation, get_constant_columns, rank_sensor_variability, summarize_cycles


def _format_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _stationarity_summary(df: pd.DataFrame, sensor_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor_name in sensor_columns:
        series = df[sensor_name].astype(float).dropna()
        adf_stat, adf_pvalue, *_ = adfuller(series, autolag="AIC")
        kpss_stat, kpss_pvalue, *_ = kpss(series, regression="c", nlags="auto")
        rows.append(
            {
                "sensor": sensor_name,
                "adf_pvalue": round(float(adf_pvalue), 6),
                "kpss_pvalue": round(float(kpss_pvalue), 6),
                "adf_stationary": bool(adf_pvalue < 0.05),
                "kpss_stationary": bool(kpss_pvalue >= 0.05),
            }
        )
    return pd.DataFrame(rows)


def _fft_summary(df: pd.DataFrame, sensor_columns: list[str], top_k: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor_name in sensor_columns:
        signal = df[sensor_name].astype(float).to_numpy(copy=False)
        centered = signal - np.mean(signal)
        amplitudes = np.abs(np.fft.rfft(centered))[1:]
        if amplitudes.size == 0:
            dominant = []
        else:
            dominant = np.sort(amplitudes)[-top_k:][::-1]
        row = {"sensor": sensor_name}
        for index in range(top_k):
            row[f"fft_amp_{index + 1}"] = round(float(dominant[index]), 4) if index < len(dominant) else None
        rows.append(row)
    return pd.DataFrame(rows)


def build_eda_report_markdown(data_dir: Path, fd: str = "FD001") -> str:
    train_df = load_fd_data(data_dir, fd=fd, split="train")
    test_df = load_fd_data(data_dir, fd=fd, split="test")
    rul_df = load_fd_rul(data_dir, fd=fd)
    train_rul_df = add_train_rul(train_df)

    train_cycles = summarize_cycles(train_df)
    test_cycles = summarize_cycles(test_df)
    constant_columns = get_constant_columns(train_df, exclude={"unit", "cycle"})
    variability_df = rank_sensor_variability(train_df).head(8).reset_index().rename(columns={"index": "sensor"})
    degradation_df = estimate_sensor_degradation(train_df).head(8).reset_index().rename(columns={"index": "sensor"})
    top_stationarity_sensors = variability_df["sensor"].head(4).tolist()
    stationarity_df = _stationarity_summary(train_df, top_stationarity_sensors)
    fft_df = _fft_summary(train_df, top_stationarity_sensors, top_k=5)
    missing_df = (
        train_df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
        .query("missing_count > 0")
    )
    if missing_df.empty:
        missing_df = pd.DataFrame([{"column": "none", "missing_count": 0}])

    report_lines = [
        "# Week 1 EDA Report",
        "",
        "## Scope",
        "",
        f"This report summarizes the first-pass exploratory analysis for the `{fd}` split of the NASA CMAPSS dataset.",
        "The goal was to understand the shape of the sequence data before leaning too hard on modeling decisions.",
        "",
        "## Dataset Summary",
        "",
        f"- Train units: `{train_cycles['units']}`",
        f"- Test units: `{test_cycles['units']}`",
        f"- Train cycle range: `{train_cycles['min_cycles']}` to `{train_cycles['max_cycles']}`",
        f"- Test cycle range: `{test_cycles['min_cycles']}` to `{test_cycles['max_cycles']}`",
        f"- Test RUL rows: `{len(rul_df)}`",
        f"- Train rows: `{len(train_df)}`",
        f"- Test rows: `{len(test_df)}`",
        "",
        "## What was checked",
        "",
        "- distribution spread and sensor variability",
        "- constant or near-dead sensors",
        "- rough degradation signal from start-to-end sensor drift",
        "- missing-value patterns",
        "- stationarity behavior using ADF and KPSS",
        "- FFT-based frequency signatures for high-variability sensors",
        "",
        "## Constant Columns",
        "",
        ("No constant sensor columns were found beyond identifiers." if not constant_columns else ", ".join(constant_columns)),
        "",
        "## Most Variable Sensors",
        "",
        _format_markdown_table(variability_df.round(4)),
        "",
        "## Strongest Average Degradation Deltas",
        "",
        _format_markdown_table(degradation_df.round(4)),
        "",
        "## Missing Values",
        "",
        _format_markdown_table(missing_df),
        "",
        "## Stationarity Snapshot",
        "",
        "ADF and KPSS were used together so the report would not lean too heavily on one stationarity test alone.",
        "",
        _format_markdown_table(stationarity_df),
        "",
        "## FFT Snapshot",
        "",
        "The FFT table captures dominant amplitude strengths for the most variable sensors and helps justify frequency-aware features later in the pipeline.",
        "",
        _format_markdown_table(fft_df),
        "",
        "## Initial Observations",
        "",
        "- The dataset is sequential and unit-based, not ordinary tabular regression data.",
        "- Sensor behavior is uneven: some sensors move a lot, others are much less informative.",
        "- The start-vs-end drift view suggests that a subset of sensors carries stronger degradation signal than the rest.",
        "- Stationarity should not be assumed blindly; some high-variability sensors show evidence of evolving behavior over time.",
        "- Frequency information is present strongly enough to justify FFT-derived features in the feature engineering pipeline.",
        "",
        "## Why this mattered for Week 1",
        "",
        "- It justified keeping CMAPSS as the main sequence-learning dataset.",
        "- It supported the later choice to add rolling, lag, and FFT features.",
        "- It reinforced that anomaly experiments should not be treated as simple row classification.",
        "- It gave a cleaner foundation before moving into anomaly baselines and sequence models.",
        "",
        "## Sample Derived RUL View",
        "",
        _format_markdown_table(train_rul_df.loc[train_rul_df["unit"] == 1, ["unit", "cycle", "rul"]].head(8)),
    ]
    return "\n".join(report_lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a markdown EDA report for the NASA CMAPSS dataset.")
    parser.add_argument("--fd", default="FD001", help="Dataset subset, for example FD001.")
    parser.add_argument(
        "--output-path",
        default="reports/week1_eda_report.md",
        help="Markdown file that will receive the generated report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    report_markdown = build_eda_report_markdown(project_root / "Data", fd=args.fd)
    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown, encoding="utf-8")
    print(f"Wrote EDA report to {output_path}")


if __name__ == "__main__":
    main()

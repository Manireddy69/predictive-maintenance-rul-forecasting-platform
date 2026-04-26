from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from .maintenance_scheduler import MaintenanceCostMatrix, SchedulerResources, build_scheduler_tasks_from_predictions, solve_maintenance_schedule
from .week2_checkpoint import find_latest_sequence_predictions


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _benchmark(function: Callable[[], object], iterations: int) -> dict[str, float | int]:
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return {
        "iterations": iterations,
        "min_seconds": min(timings),
        "median_seconds": statistics.median(timings),
        "mean_seconds": statistics.mean(timings),
        "max_seconds": max(timings),
    }


def benchmark_batch_artifact_load(iterations: int) -> dict[str, float | int | str]:
    predictions_csv = find_latest_sequence_predictions()
    if predictions_csv is None:
        raise FileNotFoundError("No saved RUL predictions were found for batch artifact benchmark.")

    def load_predictions() -> pd.DataFrame:
        return pd.read_csv(predictions_csv)

    result = _benchmark(load_predictions, iterations)
    result["mode"] = "batch_artifact_load"
    result["source"] = str(predictions_csv)
    return result


def benchmark_scheduler(iterations: int, max_tasks: int) -> dict[str, float | int | str]:
    predictions_csv = find_latest_sequence_predictions()
    if predictions_csv is None:
        raise FileNotFoundError("No saved RUL predictions were found for scheduler benchmark.")

    predictions = pd.read_csv(predictions_csv)
    resources = SchedulerResources()
    cost_matrix = MaintenanceCostMatrix()
    tasks = build_scheduler_tasks_from_predictions(
        predictions=predictions,
        planning_horizon_days=resources.planning_horizon_days,
        cost_matrix=cost_matrix,
        max_tasks=max_tasks,
    )

    def solve_once() -> object:
        return solve_maintenance_schedule(tasks=tasks, resources=resources, cost_matrix=cost_matrix)

    result = _benchmark(solve_once, iterations)
    result["mode"] = "scheduler_solve"
    result["task_count"] = len(tasks)
    result["source"] = str(predictions_csv)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local latency evidence for the capstone submission.")
    parser.add_argument(
        "--mode",
        choices=["batch-artifact-load", "scheduler-solve", "all"],
        default="all",
        help="Benchmark mode to run.",
    )
    parser.add_argument("--iterations", type=int, default=5, help="Number of timed iterations.")
    parser.add_argument("--max-tasks", type=int, default=20, help="Scheduler task cap for solve benchmark.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root() / "reports" / "latency_benchmark.json",
        help="Path for JSON benchmark evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")

    results: list[dict[str, float | int | str]] = []
    if args.mode in {"batch-artifact-load", "all"}:
        results.append(benchmark_batch_artifact_load(args.iterations))
    if args.mode in {"scheduler-solve", "all"}:
        results.append(benchmark_scheduler(args.iterations, args.max_tasks))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=== Latency benchmark ===")
    print(json.dumps(results, indent=2))
    print(f"\nJSON: {args.output_json}")


if __name__ == "__main__":
    main()

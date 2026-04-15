from __future__ import annotations

import numpy as np

from src.mlflow_tracking import compute_regression_calibration_metrics


def test_compute_regression_calibration_metrics_returns_expected_keys() -> None:
    metrics = compute_regression_calibration_metrics(
        targets=np.array([10.0, 8.0, 6.0, 4.0], dtype=float),
        predictions=np.array([9.0, 7.0, 5.0, 3.0], dtype=float),
        num_bins=2,
    )

    assert {"calibration_slope", "calibration_intercept", "calibration_mae"} == set(metrics)
    assert metrics["calibration_slope"] > 0
    assert metrics["calibration_mae"] >= 0


def test_compute_regression_calibration_metrics_handles_short_input() -> None:
    metrics = compute_regression_calibration_metrics(
        targets=np.array([5.0], dtype=float),
        predictions=np.array([5.5], dtype=float),
    )

    assert np.isnan(metrics["calibration_slope"])
    assert np.isnan(metrics["calibration_intercept"])
    assert np.isnan(metrics["calibration_mae"])

from __future__ import annotations

import numpy as np
import pandas as pd

from src.anomaly_baseline import (
    ANOMALY_FEATURE_COLUMNS,
    ANOMALY_SCORE_COLUMNS,
    build_holdout_with_injected_anomalies,
    evaluation_summary_frame,
    evaluate_anomaly_scores,
    prepare_feature_frame,
    run_scored_baseline_experiment,
)


def _make_holdout_df() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for unit in (1, 2):
        for cycle in range(1, 11):
            row: dict[str, float | int] = {
                "unit": unit,
                "cycle": cycle,
                "setting_1": 0.01 * cycle,
                "setting_2": 0.001 * cycle,
                "setting_3": 100.0,
            }
            for index in range(1, 22):
                row[f"sensor_{index}"] = 500.0 + cycle + index
            rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_feature_frame_returns_expected_columns() -> None:
    df = _make_holdout_df()

    feature_df = prepare_feature_frame(df)

    assert list(feature_df.columns) == ANOMALY_FEATURE_COLUMNS
    assert feature_df.shape == (20, len(ANOMALY_FEATURE_COLUMNS))


def test_build_holdout_with_injected_anomalies_adds_positive_rows() -> None:
    holdout_df = _make_holdout_df()

    evaluation_df = build_holdout_with_injected_anomalies(
        holdout_df,
        anomaly_fraction=0.2,
        random_state=7,
    )

    assert len(evaluation_df) > len(holdout_df)
    assert int(evaluation_df["is_anomaly"].sum()) == 4
    assert set(evaluation_df["injected_anomaly_type"].unique()) >= {"normal"}


def test_build_holdout_with_injected_anomalies_in_place_preserves_length() -> None:
    holdout_df = _make_holdout_df()

    evaluation_df = build_holdout_with_injected_anomalies(
        holdout_df,
        anomaly_fraction=0.2,
        random_state=7,
        append_anomalies=False,
    )

    assert len(evaluation_df) == len(holdout_df)
    assert int(evaluation_df["is_anomaly"].sum()) == 4


def test_evaluate_anomaly_scores_returns_perfect_metrics_for_separable_scores() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    result = evaluate_anomaly_scores(y_true, scores, model_name="demo")

    assert result.roc_auc == 1.0
    assert result.pr_auc == 1.0
    assert len(result.precision) == len(result.recall)
    assert len(result.thresholds) == len(result.precision) - 1


def test_run_scored_baseline_experiment_returns_plot_ready_scores() -> None:
    train_df = _make_holdout_df()
    holdout_df = _make_holdout_df()

    evaluation_df, results = run_scored_baseline_experiment(
        train_df=train_df,
        holdout_df=holdout_df,
        anomaly_fraction=0.2,
        contamination=0.2,
        random_state=7,
        n_neighbors=5,
    )

    assert len(evaluation_df) > len(holdout_df)
    assert set(ANOMALY_SCORE_COLUMNS).issubset(evaluation_df.columns)
    for score_column in ANOMALY_SCORE_COLUMNS:
        assert np.issubdtype(evaluation_df[score_column].dtype, np.number)
        assert evaluation_df[score_column].notna().all()

    assert set(results) == {"isolation_forest", "local_outlier_factor", "zscore", "mad"}


def test_evaluation_summary_frame_orders_best_pr_auc_first() -> None:
    train_df = _make_holdout_df()
    holdout_df = _make_holdout_df()
    _, results = run_scored_baseline_experiment(
        train_df=train_df,
        holdout_df=holdout_df,
        anomaly_fraction=0.2,
        contamination=0.2,
        random_state=7,
        n_neighbors=5,
    )

    summary_df = evaluation_summary_frame(results)

    assert list(summary_df.columns) == ["model_key", "model_name", "roc_auc", "pr_auc", "pr_curve_points"]
    assert summary_df["pr_auc"].is_monotonic_decreasing

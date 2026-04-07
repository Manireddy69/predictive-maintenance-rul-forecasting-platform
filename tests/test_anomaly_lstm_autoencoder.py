from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from src.anomaly_lstm_autoencoder import (
    LSTMAutoencoderConfig,
    aggregate_window_scores,
    build_sequence_windows,
    fit_sequence_scaler,
    reconstruction_error_threshold,
    save_comparison_artifacts,
    transform_sequence_windows,
)


def _make_sequence_df() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for unit in (1, 2):
        for cycle in range(1, 7):
            row: dict[str, float | int] = {
                "unit": unit,
                "cycle": cycle,
                "setting_1": 0.01 * cycle,
                "setting_2": 0.001 * cycle,
                "setting_3": 100.0,
                "is_anomaly": 1 if unit == 2 and cycle >= 5 else 0,
                "row_score": float(cycle),
            }
            for sensor_index in range(1, 22):
                row[f"sensor_{sensor_index}"] = 500.0 + unit + cycle + sensor_index
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_sequence_windows_tracks_window_labels() -> None:
    df = _make_sequence_df()

    window_set = build_sequence_windows(
        df,
        sequence_length=3,
        stride=2,
        label_column="is_anomaly",
    )

    assert window_set.windows.shape == (4, 3, 24)
    assert window_set.metadata["is_anomaly_window"].tolist() == [0, 0, 0, 1]
    assert window_set.metadata["cycle_start"].tolist() == [1, 3, 1, 3]
    assert window_set.metadata["cycle_end"].tolist() == [3, 5, 3, 5]


def test_sequence_scaler_round_trips_shape() -> None:
    df = _make_sequence_df()
    window_set = build_sequence_windows(df, sequence_length=3, stride=1)

    scaler = fit_sequence_scaler(window_set.windows)
    transformed = transform_sequence_windows(window_set.windows, scaler)

    assert transformed.shape == window_set.windows.shape
    flattened = transformed.reshape(-1, transformed.shape[-1])
    assert np.allclose(flattened.mean(axis=0), 0.0, atol=1e-7)


def test_aggregate_window_scores_uses_window_max() -> None:
    df = _make_sequence_df()

    aggregated = aggregate_window_scores(
        df,
        score_column="row_score",
        sequence_length=3,
        stride=2,
    )

    assert aggregated["row_score"].tolist() == [3.0, 5.0, 3.0, 5.0]


def test_reconstruction_error_threshold_uses_upper_quantile() -> None:
    errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

    threshold = reconstruction_error_threshold(errors, contamination=0.2)

    assert np.isclose(threshold, 0.42)


def test_save_comparison_artifacts_writes_expected_files() -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "model_key": "mad",
                "model_name": "MAD Distance",
                "roc_auc": 0.94,
                "pr_auc": 0.93,
                "pr_curve_points": 100,
            }
        ]
    )
    combined_window_scores = pd.DataFrame(
        [{"unit": 1, "window_start_index": 0, "window_end_index": 2, "is_anomaly_window": 0, "reconstruction_error": 0.1}]
    )
    scored_holdout_df = pd.DataFrame([{"unit": 1, "cycle": 1, "mad_score": 0.2, "is_anomaly": 0}])
    history_df = pd.DataFrame([{"epoch": 1, "train_loss": 0.5}])

    output_root = Path("Data") / "test_artifacts" / f"save_check_{uuid.uuid4().hex}"
    try:
        artifacts = save_comparison_artifacts(
            comparison_df=comparison_df,
            combined_window_scores=combined_window_scores,
            scored_holdout_df=scored_holdout_df,
            history_df=history_df,
            threshold=0.56,
            device_name="cpu",
            output_root=output_root,
            run_name="demo_run",
            config=LSTMAutoencoderConfig(num_epochs=1),
            anomaly_fraction=0.08,
            random_state=42,
        )

        assert artifacts.output_dir.exists()
        assert artifacts.comparison_csv.exists()
        assert artifacts.window_scores_csv.exists()
        assert artifacts.row_scores_csv.exists()
        assert artifacts.training_history_csv.exists()
        assert artifacts.metadata_json.exists()
        assert artifacts.summary_markdown.exists()
    finally:
        shutil.rmtree(output_root, ignore_errors=True)

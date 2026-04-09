from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.sequence_attention_model import (
    FitResult,
    OptunaSearchConfig,
    SequenceModelConfig,
    SequenceTrainingConfig,
    BidirectionalAttentionSequenceModel,
    apply_trial_parameters,
    build_prediction_frame,
    compute_task_metrics,
    optuna_objective,
    suggest_hyperparameters,
)


def _make_train_df(num_units: int = 4, cycles_per_unit: int = 8) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for unit in range(1, num_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            row: dict[str, float | int] = {
                "unit": unit,
                "cycle": cycle,
                "setting_1": 0.1 * unit,
                "setting_2": 0.01 * cycle,
                "setting_3": 100.0,
                "rul": float(cycles_per_unit - cycle),
            }
            for sensor_idx in range(1, 22):
                row[f"sensor_{sensor_idx}"] = float(unit * 10 + cycle + sensor_idx)
            rows.append(row)
    return pd.DataFrame(rows)


class FakeTrial:
    def __init__(self) -> None:
        self.number = 3

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        values = {
            "learning_rate": 5e-4,
            "dropout": 0.24,
        }
        return values[name]

    def suggest_categorical(self, name: str, choices: list[int]) -> int:
        values = {
            "hidden_size": choices[-1],
            "window_size": choices[0],
        }
        return values[name]

    def suggest_int(self, name: str, low: int, high: int) -> int:
        assert (low, high) == (2, 3)
        return 3


def test_bidirectional_attention_model_returns_sequence_attention_weights() -> None:
    model = BidirectionalAttentionSequenceModel(
        input_dim=24,
        target_mode="rul",
        hidden_size=32,
        num_layers=2,
        dropout=0.25,
    )
    inputs = torch.randn(5, 7, 24)

    outputs, attention = model(inputs)

    assert outputs.shape == (5,)
    assert attention.shape == (5, 7)
    assert torch.allclose(attention.sum(dim=1), torch.ones(5), atol=1e-6)


def test_compute_task_metrics_supports_regression_and_classification() -> None:
    regression_metrics = compute_task_metrics(
        target_mode="rul",
        targets=np.array([4.0, 2.0, 1.0], dtype=np.float32),
        predictions=np.array([3.0, 2.5, 1.5], dtype=np.float32),
    )
    classification_metrics = compute_task_metrics(
        target_mode="failure_in_next_window",
        targets=np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        predictions=np.array([0.1, 0.8, 0.6, 0.3], dtype=np.float32),
    )

    assert regression_metrics["rmse"] > 0.0
    assert regression_metrics["mae"] > 0.0
    assert "r2" in regression_metrics
    assert classification_metrics["accuracy"] == 1.0
    assert classification_metrics["roc_auc"] == 1.0
    assert classification_metrics["pr_auc"] == 1.0


def test_build_prediction_frame_adds_attention_summaries() -> None:
    metadata = pd.DataFrame({"unit": [1, 2], "cycle_end": [30, 40]})
    prediction_frame = build_prediction_frame(
        metadata=metadata,
        target_mode="failure_in_next_window",
        target_name="failure_in_next_window",
        targets=np.array([0.0, 1.0], dtype=np.float32),
        predictions=np.array([0.2, 0.7], dtype=np.float32),
        attention_weights=np.array([[0.1, 0.3, 0.6], [0.5, 0.2, 0.3]], dtype=np.float32),
    )

    assert prediction_frame["attention_peak_index"].tolist() == [2, 0]
    assert prediction_frame["predicted_label"].tolist() == [0, 1]
    assert np.allclose(prediction_frame["predicted_probability"].to_numpy(), np.array([0.2, 0.7]))


def test_suggest_hyperparameters_matches_day9_search_ranges() -> None:
    trial = FakeTrial()
    params = suggest_hyperparameters(trial, OptunaSearchConfig())

    assert params["hidden_size"] == 256
    assert params["window_size"] == 20
    assert 0.2 <= params["dropout"] <= 0.3
    assert params["num_layers"] in {2, 3}


def test_apply_trial_parameters_updates_model_and_window_size() -> None:
    base_model_config = SequenceModelConfig()
    base_training_config = SequenceTrainingConfig()

    updated_model_config, updated_training_config = apply_trial_parameters(
        model_config=base_model_config,
        training_config=base_training_config,
        trial_params={
            "hidden_size": 192,
            "dropout": 0.22,
            "learning_rate": 7e-4,
            "num_layers": 3,
            "window_size": 40,
        },
    )

    assert updated_model_config.hidden_size == 192
    assert updated_model_config.dropout == 0.22
    assert updated_model_config.learning_rate == 7e-4
    assert updated_model_config.num_layers == 3
    assert updated_training_config.window_size == 40


def test_optuna_objective_uses_trial_parameters(monkeypatch) -> None:
    captured: dict[str, float | int] = {}

    def fake_fit_sequence_model(datamodule, model_config, training_config, output_dir, enable_checkpointing):
        captured["window_size"] = datamodule.window_size
        captured["hidden_size"] = model_config.hidden_size
        captured["dropout"] = model_config.dropout
        captured["num_layers"] = model_config.num_layers
        captured["learning_rate"] = model_config.learning_rate
        captured["output_dir_name"] = output_dir.name
        return FitResult(model=object(), monitor_name="val_rmse", monitor_value=0.123, checkpoint_path=None)

    monkeypatch.setattr("src.sequence_attention_model.fit_sequence_model", fake_fit_sequence_model)

    result = optuna_objective(
        trial=FakeTrial(),
        train_df=_make_train_df(),
        base_model_config=SequenceModelConfig(),
        base_training_config=SequenceTrainingConfig(target_mode="rul", window_size=30),
        output_dir=Path("tests"),
    )

    assert result == 0.123
    assert captured["window_size"] == 20
    assert captured["hidden_size"] == 256
    assert captured["num_layers"] == 3
    assert captured["output_dir_name"] == "trial_3"

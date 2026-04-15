from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.sequence_attention_model import (
    FitResult,
    OptunaSearchConfig,
    ProphetEnsembleConfig,
    ProphetForecast,
    SequenceModelConfig,
    SequenceTrainingConfig,
    BidirectionalAttentionSequenceModel,
    apply_trial_parameters,
    build_prediction_frame,
    build_prophet_training_frame,
    build_weighted_rul_ensemble,
    compute_task_metrics,
    evaluate_model,
    forecast_prophet_rul_with_timeout,
    fit_prophet_rul_baseline,
    optuna_objective,
    predict_prophet_rul,
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


class FakeProphetModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.fit_frame: pd.DataFrame | None = None
        self.seasonality_name = "cycle_seasonality"
        self.added_seasonality: dict[str, float | int] | None = None

    def add_seasonality(self, name: str, period: float, fourier_order: int) -> None:
        self.seasonality_name = name
        self.added_seasonality = {"name": name, "period": period, "fourier_order": fourier_order}

    def fit(self, frame: pd.DataFrame) -> "FakeProphetModel":
        self.fit_frame = frame.copy()
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = len(frame)
        return pd.DataFrame(
            {
                "ds": frame["ds"],
                "yhat": np.linspace(20.0, 20.0 + rows - 1, rows, dtype=np.float32),
                "trend": np.linspace(18.0, 18.0 + rows - 1, rows, dtype=np.float32),
                self.seasonality_name: np.full(rows, 2.0, dtype=np.float32),
            }
        )


class DummySequenceDataset:
    def __init__(self, targets: list[float]) -> None:
        self.targets = torch.tensor(targets, dtype=torch.float32)


class DummySequenceDataModule:
    def __init__(self) -> None:
        self.train_dataset = DummySequenceDataset([10.0, 9.0, 8.0])
        self.train_metadata = pd.DataFrame({"target_cycle": [1, 2, 3]})
        self.test_dataset = object()
        self.test_metadata = pd.DataFrame({"unit": [1, 2], "target_cycle": [2, 3]})
        self.target_name_ = "rul"

    def test_dataloader(self) -> str:
        return "test_loader"


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


def test_day9_sequence_model_config_enforces_dropout_and_layer_constraints() -> None:
    with pytest.raises(ValueError, match="dropout"):
        SequenceModelConfig(dropout=0.19)

    with pytest.raises(ValueError, match="2 or 3"):
        SequenceModelConfig(num_layers=4)


def test_optuna_search_config_stays_within_day9_dropout_and_layer_bounds() -> None:
    with pytest.raises(ValueError, match="dropout"):
        OptunaSearchConfig(dropout_low=0.15, dropout_high=0.3)

    with pytest.raises(ValueError, match="2 or 3"):
        OptunaSearchConfig(num_layers_choices=(1, 2))


def test_prophet_ensemble_config_requires_weights_to_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        ProphetEnsembleConfig(lstm_weight=0.8, prophet_weight=0.3)


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


def test_build_prophet_training_frame_aggregates_window_targets_by_cycle() -> None:
    metadata = pd.DataFrame({"target_cycle": [1, 1, 2, 3]})
    targets = np.array([12.0, 14.0, 9.0, 7.0], dtype=np.float32)

    training_frame = build_prophet_training_frame(metadata=metadata, targets=targets)

    assert training_frame["target_cycle"].tolist() == [1, 2, 3]
    assert np.allclose(training_frame["y"].to_numpy(), np.array([13.0, 9.0, 7.0], dtype=np.float32))
    assert list(training_frame.columns) == ["ds", "target_cycle", "y"]


def test_fit_and_predict_prophet_rul_baseline_without_real_dependency(monkeypatch) -> None:
    fake_prophet_instances: list[FakeProphetModel] = []

    def fake_prophet_factory(**kwargs) -> FakeProphetModel:
        instance = FakeProphetModel(**kwargs)
        fake_prophet_instances.append(instance)
        return instance

    monkeypatch.setattr("src.sequence_attention_model._load_prophet_runtime", lambda: fake_prophet_factory)

    baseline = fit_prophet_rul_baseline(
        metadata=pd.DataFrame({"target_cycle": [1, 1, 2, 3]}),
        targets=np.array([12.0, 14.0, 9.0, 7.0], dtype=np.float32),
        config=ProphetEnsembleConfig(),
    )
    forecast = predict_prophet_rul(
        baseline=baseline,
        metadata=pd.DataFrame({"target_cycle": [2, 4]}),
    )

    assert fake_prophet_instances, "Expected the fake Prophet runtime to be used."
    assert fake_prophet_instances[0].fit_frame is not None
    assert fake_prophet_instances[0].added_seasonality == {
        "name": "cycle_seasonality",
        "period": 20.0,
        "fourier_order": 5,
    }
    assert np.allclose(forecast.predictions, np.array([20.0, 21.0], dtype=np.float32))
    assert np.allclose(forecast.trend, np.array([18.0, 19.0], dtype=np.float32))
    assert np.allclose(forecast.seasonality, np.array([2.0, 2.0], dtype=np.float32))


def test_forecast_prophet_rul_with_timeout_reads_worker_output(monkeypatch) -> None:
    temp_dir = Path("Data/test_artifacts/prophet_worker_unit_test").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)

    class FakeTemporaryDirectory:
        def __init__(self, *args, **kwargs) -> None:
            self.path = temp_dir

        def __enter__(self) -> str:
            return str(self.path)

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_subprocess_run(command, cwd, capture_output, text, check, timeout):
        assert command[:3] == [__import__("sys").executable, "-m", "src.prophet_worker"]
        output_json = Path(command[command.index("--output-json") + 1])
        output_json.write_text(
            '{"predictions": [11.0, 12.0], "trend": [9.0, 10.0], "seasonality": [2.0, 2.0]}',
            encoding="utf-8",
        )

        class Result:
            stdout = ""
            stderr = ""
            returncode = 0

        return Result()

    monkeypatch.setattr("src.sequence_attention_model.tempfile.TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr("src.sequence_attention_model.subprocess.run", fake_subprocess_run)

    forecast = forecast_prophet_rul_with_timeout(
        training_metadata=pd.DataFrame({"target_cycle": [1, 2, 3]}),
        training_targets=np.array([12.0, 10.0, 8.0], dtype=np.float32),
        scoring_metadata=pd.DataFrame({"target_cycle": [4, 5]}),
        config=ProphetEnsembleConfig(),
    )

    assert np.allclose(forecast.predictions, np.array([11.0, 12.0], dtype=np.float32))
    assert np.allclose(forecast.trend, np.array([9.0, 10.0], dtype=np.float32))
    assert np.allclose(forecast.seasonality, np.array([2.0, 2.0], dtype=np.float32))


def test_build_weighted_rul_ensemble_respects_requested_mix() -> None:
    ensemble = build_weighted_rul_ensemble(
        lstm_predictions=np.array([10.0, 6.0], dtype=np.float32),
        prophet_predictions=np.array([8.0, 8.0], dtype=np.float32),
        config=ProphetEnsembleConfig(lstm_weight=0.7, prophet_weight=0.3),
    )

    assert np.allclose(ensemble, np.array([9.4, 6.6], dtype=np.float32))


def test_evaluate_model_adds_prophet_components_to_rul_predictions(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.sequence_attention_model.predict_from_dataloader",
        lambda model, dataloader, target_mode: (
            np.array([10.0, 6.0], dtype=np.float32),
            np.array([9.0, 7.0], dtype=np.float32),
            np.array([[0.7, 0.3], [0.6, 0.4]], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        "src.sequence_attention_model.forecast_prophet_rul_with_timeout",
        lambda training_metadata, training_targets, scoring_metadata, config, cycle_column="target_cycle": ProphetForecast(
            predictions=np.array([8.0, 8.0], dtype=np.float32),
            trend=np.array([7.5, 7.0], dtype=np.float32),
            seasonality=np.array([0.5, 1.0], dtype=np.float32),
        ),
    )

    metrics, prediction_frame = evaluate_model(
        model=object(),
        datamodule=DummySequenceDataModule(),
        target_mode="rul",
        prophet_config=ProphetEnsembleConfig(),
    )

    assert "hybrid_rmse" in metrics
    assert "lstm_rmse" in metrics
    assert "prophet_rmse" in metrics
    assert np.allclose(prediction_frame["lstm_prediction"].to_numpy(), np.array([10.0, 6.0], dtype=np.float32))
    assert np.allclose(prediction_frame["prophet_prediction"].to_numpy(), np.array([8.0, 8.0], dtype=np.float32))
    assert np.allclose(prediction_frame["hybrid_prediction"].to_numpy(), np.array([9.4, 6.6], dtype=np.float32))
    assert np.allclose(prediction_frame["prediction"].to_numpy(), np.array([9.4, 6.6], dtype=np.float32))


def test_evaluate_model_falls_back_to_lstm_when_prophet_worker_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.sequence_attention_model.predict_from_dataloader",
        lambda model, dataloader, target_mode: (
            np.array([10.0, 6.0], dtype=np.float32),
            np.array([9.0, 7.0], dtype=np.float32),
            np.array([[0.7, 0.3], [0.6, 0.4]], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        "src.sequence_attention_model.forecast_prophet_rul_with_timeout",
        lambda training_metadata, training_targets, scoring_metadata, config, cycle_column="target_cycle": (_ for _ in ()).throw(
            RuntimeError("Prophet stalled")
        ),
    )

    metrics, prediction_frame = evaluate_model(
        model=object(),
        datamodule=DummySequenceDataModule(),
        target_mode="rul",
        prophet_config=ProphetEnsembleConfig(),
    )

    assert metrics["ensemble_applied"] == 0.0
    assert metrics["prophet_fallback_used"] == 1.0
    assert prediction_frame["prophet_prediction"].isna().all()
    assert np.allclose(prediction_frame["prediction"].to_numpy(), np.array([10.0, 6.0], dtype=np.float32))


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

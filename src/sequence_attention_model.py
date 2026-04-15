from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

from .data import load_dataset
from .sequence_data import CMAPSSSequenceDataModule, TargetMode

try:  # pragma: no cover - import path depends on local installation
    from lightning.pytorch import LightningModule, Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    LIGHTNING_AVAILABLE = True
except ImportError:  # pragma: no cover - import path depends on local installation
    try:
        from pytorch_lightning import LightningModule, Trainer, seed_everything
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

        LIGHTNING_AVAILABLE = True
    except ImportError:  # pragma: no cover - lightweight fallback for local tests
        LIGHTNING_AVAILABLE = False

        class LightningModule(nn.Module):  # type: ignore[no-redef]
            def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
                return None

            def log(self, *args: Any, **kwargs: Any) -> None:
                return None

        Trainer = Any  # type: ignore[assignment]
        EarlyStopping = object  # type: ignore[assignment]
        ModelCheckpoint = object  # type: ignore[assignment]

        def seed_everything(seed: int, workers: bool = False) -> int:  # type: ignore[no-redef]
            return seed


@dataclass(frozen=True)
class SequenceModelConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if self.num_layers not in {2, 3}:
            raise ValueError("Day 9 sequence training supports only 2 or 3 bidirectional LSTM layers.")
        if not 0.2 <= self.dropout <= 0.3:
            raise ValueError("Day 9 sequence training requires dropout to stay in the range [0.2, 0.3].")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")


@dataclass(frozen=True)
class SequenceTrainingConfig:
    target_mode: TargetMode = "rul"
    window_size: int = 30
    stride: int = 1
    prediction_horizon: int = 30
    batch_size: int = 128
    max_epochs: int = 15
    patience: int = 4
    validation_fraction: float = 0.2
    num_workers: int = 0
    random_state: int = 42
    accelerator: str = "auto"


@dataclass(frozen=True)
class OptunaSearchConfig:
    n_trials: int = 10
    learning_rate_low: float = 1e-4
    learning_rate_high: float = 5e-3
    hidden_size_choices: tuple[int, ...] = (64, 128, 192, 256)
    window_size_choices: tuple[int, ...] = (20, 30, 40, 50)
    dropout_low: float = 0.2
    dropout_high: float = 0.3
    num_layers_choices: tuple[int, ...] = (2, 3)

    def __post_init__(self) -> None:
        if self.n_trials < 0:
            raise ValueError("n_trials must be non-negative.")
        if self.learning_rate_low <= 0 or self.learning_rate_high <= 0:
            raise ValueError("Optuna learning-rate bounds must be positive.")
        if self.learning_rate_low > self.learning_rate_high:
            raise ValueError("learning_rate_low must be less than or equal to learning_rate_high.")
        if not self.hidden_size_choices:
            raise ValueError("hidden_size_choices must include at least one value.")
        if any(choice < 1 for choice in self.hidden_size_choices):
            raise ValueError("hidden_size_choices must all be positive.")
        if not self.window_size_choices:
            raise ValueError("window_size_choices must include at least one value.")
        if any(choice < 2 for choice in self.window_size_choices):
            raise ValueError("window_size_choices must all be at least 2.")
        if not 0.2 <= self.dropout_low <= self.dropout_high <= 0.3:
            raise ValueError("Day 9 Optuna dropout search must stay within the range [0.2, 0.3].")
        if not self.num_layers_choices:
            raise ValueError("num_layers_choices must include at least one value.")
        if any(choice not in {2, 3} for choice in self.num_layers_choices):
            raise ValueError("Day 9 Optuna layer choices must be limited to 2 or 3.")


@dataclass(frozen=True)
class ProphetEnsembleConfig:
    enabled: bool = True
    lstm_weight: float = 0.7
    prophet_weight: float = 0.3
    seasonality_period: float = 20.0
    seasonality_fourier_order: int = 5
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    fit_timeout_seconds: int = 60
    fallback_to_lstm_on_failure: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.lstm_weight <= 1.0:
            raise ValueError("lstm_weight must be in the range [0, 1].")
        if not 0.0 <= self.prophet_weight <= 1.0:
            raise ValueError("prophet_weight must be in the range [0, 1].")
        if not math.isclose(self.lstm_weight + self.prophet_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("lstm_weight and prophet_weight must sum to 1.0.")
        if self.seasonality_period <= 1:
            raise ValueError("seasonality_period must be greater than 1.")
        if self.seasonality_fourier_order < 1:
            raise ValueError("seasonality_fourier_order must be positive.")
        if self.changepoint_prior_scale <= 0:
            raise ValueError("changepoint_prior_scale must be positive.")
        if self.seasonality_prior_scale <= 0:
            raise ValueError("seasonality_prior_scale must be positive.")
        if self.fit_timeout_seconds < 1:
            raise ValueError("fit_timeout_seconds must be positive.")


@dataclass(frozen=True)
class FitResult:
    model: "BidirectionalAttentionSequenceModel"
    monitor_name: str
    monitor_value: float
    checkpoint_path: str | None


@dataclass(frozen=True)
class HyperparameterSearchResult:
    best_model_config: SequenceModelConfig
    best_training_config: SequenceTrainingConfig
    best_value: float
    trials_frame: pd.DataFrame


@dataclass(frozen=True)
class SavedSequenceArtifacts:
    output_dir: Path
    metrics_json: Path
    predictions_csv: Path
    metadata_json: Path
    summary_markdown: Path
    trials_csv: Path | None = None
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class ProphetBaseline:
    model: Any
    training_frame: pd.DataFrame
    config: ProphetEnsembleConfig
    seasonality_column: str


@dataclass(frozen=True)
class ProphetForecast:
    predictions: np.ndarray
    trend: np.ndarray
    seasonality: np.ndarray


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        attention_dim = max(input_dim // 2, 1)
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, sequence_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_scores = self.score(sequence_outputs).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), sequence_outputs).squeeze(1)
        return context, attention_weights


class BidirectionalAttentionSequenceModel(LightningModule):
    def __init__(
        self,
        input_dim: int,
        target_mode: TargetMode = "rul",
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()

        if target_mode not in {"rul", "failure_in_next_window"}:
            raise ValueError(f"Unsupported target_mode: {target_mode}")
        if num_layers not in {2, 3}:
            raise ValueError("Day 9 sequence training supports only 2 or 3 bidirectional LSTM layers.")
        if not 0.2 <= dropout <= 0.3:
            raise ValueError("Day 9 sequence training requires dropout to stay in the range [0.2, 0.3].")

        self.save_hyperparameters()
        self.target_mode = target_mode
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        encoded_dim = hidden_size * 2
        self.attention = TemporalAttention(encoded_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(encoded_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_outputs, (hidden_state, _) = self.encoder(inputs)
        context_vector, attention_weights = self.attention(sequence_outputs)
        final_state = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)
        combined = torch.cat([context_vector, final_state], dim=1)
        outputs = self.head(self.dropout(combined)).squeeze(-1)
        return outputs, attention_weights

    def _loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "rul":
            return nn.functional.mse_loss(predictions, targets)
        return nn.functional.binary_cross_entropy_with_logits(predictions, targets)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        features, targets = batch
        targets = targets.float()
        predictions, _ = self(features)
        loss = self._loss(predictions, targets)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=stage != "train", batch_size=features.size(0))

        if stage != "train" and self.target_mode == "rul":
            rmse = torch.sqrt(nn.functional.mse_loss(predictions, targets))
            self.log("val_rmse", rmse, on_epoch=True, prog_bar=True, batch_size=features.size(0))
        elif stage != "train":
            probabilities = torch.sigmoid(predictions)
            accuracy = ((probabilities >= 0.5).float() == targets).float().mean()
            self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True, batch_size=features.size(0))

        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


def _load_optuna_runtime() -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError("Optuna is not installed. Add `optuna` to the environment before running tuning.") from exc
    return optuna


def _load_prophet_runtime() -> Any:
    try:
        from prophet import Prophet
    except ImportError:  # pragma: no cover - depends on local installation
        try:
            from fbprophet import Prophet  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - depends on local installation
            raise RuntimeError(
                "Prophet is not installed. Add `prophet` to the environment before running the hybrid ensemble."
            ) from exc
    return Prophet


def monitor_name_for_target_mode(target_mode: TargetMode) -> str:
    return "val_rmse" if target_mode == "rul" else "val_loss"


def monitor_mode_for_target_mode(target_mode: TargetMode) -> str:
    return "min"


def _metric_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def suggest_hyperparameters(trial: Any, search_config: OptunaSearchConfig) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            search_config.learning_rate_low,
            search_config.learning_rate_high,
            log=True,
        ),
        "hidden_size": trial.suggest_categorical("hidden_size", list(search_config.hidden_size_choices)),
        "window_size": trial.suggest_categorical("window_size", list(search_config.window_size_choices)),
        "dropout": trial.suggest_float("dropout", search_config.dropout_low, search_config.dropout_high),
        "num_layers": trial.suggest_int(
            "num_layers",
            min(search_config.num_layers_choices),
            max(search_config.num_layers_choices),
        ),
    }


def apply_trial_parameters(
    model_config: SequenceModelConfig,
    training_config: SequenceTrainingConfig,
    trial_params: dict[str, Any],
) -> tuple[SequenceModelConfig, SequenceTrainingConfig]:
    updated_model_config = replace(
        model_config,
        hidden_size=int(trial_params.get("hidden_size", model_config.hidden_size)),
        dropout=float(trial_params.get("dropout", model_config.dropout)),
        learning_rate=float(trial_params.get("learning_rate", model_config.learning_rate)),
        num_layers=int(trial_params.get("num_layers", model_config.num_layers)),
    )
    updated_training_config = replace(
        training_config,
        window_size=int(trial_params.get("window_size", training_config.window_size)),
    )
    return updated_model_config, updated_training_config


def build_datamodule(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    test_rul_df: pd.DataFrame | None,
    training_config: SequenceTrainingConfig,
) -> CMAPSSSequenceDataModule:
    return CMAPSSSequenceDataModule(
        train_df=train_df,
        test_df=test_df,
        test_rul_df=test_rul_df,
        target_mode=training_config.target_mode,
        window_size=training_config.window_size,
        stride=training_config.stride,
        prediction_horizon=training_config.prediction_horizon,
        validation_fraction=training_config.validation_fraction,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        random_state=training_config.random_state,
    )


def _cycles_to_prophet_ds(cycles: pd.Series | np.ndarray) -> pd.Series:
    cycle_values = np.asarray(cycles, dtype=np.int64)
    if cycle_values.size == 0:
        return pd.Series(dtype="datetime64[ns]")
    if np.any(cycle_values < 1):
        raise ValueError("Cycle values must be at least 1 for Prophet time-axis conversion.")
    base_timestamp = pd.Timestamp("2000-01-01")
    return pd.Series(base_timestamp + pd.to_timedelta(cycle_values - 1, unit="D"))


def build_prophet_training_frame(
    metadata: pd.DataFrame,
    targets: np.ndarray,
    cycle_column: str = "target_cycle",
) -> pd.DataFrame:
    if cycle_column not in metadata.columns:
        raise ValueError(f"Cannot build a Prophet training frame without `{cycle_column}` metadata.")
    if len(metadata) != len(targets):
        raise ValueError("metadata and targets must contain the same number of rows.")

    frame = pd.DataFrame(
        {
            cycle_column: metadata[cycle_column].to_numpy(dtype=np.int64, copy=False),
            "y": np.asarray(targets, dtype=np.float32),
        }
    )
    aggregated = frame.groupby(cycle_column, as_index=False)["y"].mean().sort_values(cycle_column).reset_index(drop=True)
    aggregated["ds"] = _cycles_to_prophet_ds(aggregated[cycle_column].to_numpy())
    return aggregated.loc[:, ["ds", cycle_column, "y"]]


def fit_prophet_rul_baseline(
    metadata: pd.DataFrame,
    targets: np.ndarray,
    config: ProphetEnsembleConfig,
    cycle_column: str = "target_cycle",
) -> ProphetBaseline:
    Prophet = _load_prophet_runtime()
    training_frame = build_prophet_training_frame(metadata=metadata, targets=targets, cycle_column=cycle_column)
    if len(training_frame) < 2:
        raise ValueError("Prophet training requires at least two aggregated cycle observations.")

    seasonality_column = "cycle_seasonality"
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=config.changepoint_prior_scale,
        seasonality_prior_scale=config.seasonality_prior_scale,
    )
    model.add_seasonality(
        name=seasonality_column,
        period=config.seasonality_period,
        fourier_order=config.seasonality_fourier_order,
    )
    model.fit(training_frame.loc[:, ["ds", "y"]])
    return ProphetBaseline(
        model=model,
        training_frame=training_frame,
        config=config,
        seasonality_column=seasonality_column,
    )


def predict_prophet_rul(
    baseline: ProphetBaseline,
    metadata: pd.DataFrame,
    cycle_column: str = "target_cycle",
) -> ProphetForecast:
    if cycle_column not in metadata.columns:
        raise ValueError(f"Cannot score Prophet predictions without `{cycle_column}` metadata.")

    forecast_input = pd.DataFrame({"ds": _cycles_to_prophet_ds(metadata[cycle_column].to_numpy(dtype=np.int64))})
    forecast = baseline.model.predict(forecast_input)
    seasonality_column = (
        baseline.seasonality_column if baseline.seasonality_column in forecast.columns else "additive_terms"
    )
    return ProphetForecast(
        predictions=forecast["yhat"].to_numpy(dtype=np.float32, copy=False),
        trend=forecast["trend"].to_numpy(dtype=np.float32, copy=False),
        seasonality=forecast[seasonality_column].to_numpy(dtype=np.float32, copy=False),
    )


def forecast_prophet_rul_with_timeout(
    training_metadata: pd.DataFrame,
    training_targets: np.ndarray,
    scoring_metadata: pd.DataFrame,
    config: ProphetEnsembleConfig,
    cycle_column: str = "target_cycle",
) -> ProphetForecast:
    training_frame = build_prophet_training_frame(
        metadata=training_metadata,
        targets=training_targets,
        cycle_column=cycle_column,
    )
    if len(training_frame) < 2:
        raise ValueError("Prophet training requires at least two aggregated cycle observations.")
    if cycle_column not in scoring_metadata.columns:
        raise ValueError(f"Cannot score Prophet predictions without `{cycle_column}` metadata.")

    project_root = Path(__file__).resolve().parent.parent
    temp_root = project_root / "Data" / "test_artifacts" / "prophet_worker"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prophet_rul_", dir=temp_root) as tmp_dir:
        tmp_path = Path(tmp_dir)
        training_csv = tmp_path / "training.csv"
        scoring_csv = tmp_path / "scoring.csv"
        config_json = tmp_path / "config.json"
        output_json = tmp_path / "forecast.json"

        training_frame.loc[:, ["ds", "y"]].to_csv(training_csv, index=False)
        pd.DataFrame({"ds": _cycles_to_prophet_ds(scoring_metadata[cycle_column].to_numpy(dtype=np.int64))}).to_csv(
            scoring_csv,
            index=False,
        )
        config_json.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

        command = [
            sys.executable,
            "-m",
            "src.prophet_worker",
            "--train-csv",
            str(training_csv),
            "--score-csv",
            str(scoring_csv),
            "--config-json",
            str(config_json),
            "--output-json",
            str(output_json),
            "--seasonality-column",
            "cycle_seasonality",
        ]
        try:
            subprocess.run(
                command,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                check=True,
                timeout=config.fit_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Prophet fit exceeded the configured timeout of {config.fit_timeout_seconds} seconds."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            message = stderr or stdout or "Prophet worker failed without additional diagnostics."
            raise RuntimeError(f"Prophet worker failed: {message}") from exc

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        return ProphetForecast(
            predictions=np.asarray(payload["predictions"], dtype=np.float32),
            trend=np.asarray(payload["trend"], dtype=np.float32),
            seasonality=np.asarray(payload["seasonality"], dtype=np.float32),
        )


def _build_prophet_or_fallback_predictions(
    *,
    lstm_predictions: np.ndarray,
    targets: np.ndarray,
    training_metadata: pd.DataFrame,
    training_targets: np.ndarray,
    scoring_metadata: pd.DataFrame,
    prophet_config: ProphetEnsembleConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    try:
        prophet_forecast = forecast_prophet_rul_with_timeout(
            training_metadata=training_metadata,
            training_targets=training_targets,
            scoring_metadata=scoring_metadata,
            config=prophet_config,
        )
    except Exception:
        if not prophet_config.fallback_to_lstm_on_failure:
            raise

        fallback_metrics = compute_task_metrics("rul", targets=targets, predictions=lstm_predictions)
        fallback_metrics.update(
            {
                "ensemble_applied": 0.0,
                "prophet_fallback_used": 1.0,
            }
        )
        fallback_frame = pd.DataFrame(
            {
                "lstm_prediction": lstm_predictions,
                "prophet_prediction": np.full_like(lstm_predictions, np.nan, dtype=np.float32),
                "prophet_trend": np.full_like(lstm_predictions, np.nan, dtype=np.float32),
                "prophet_seasonality": np.full_like(lstm_predictions, np.nan, dtype=np.float32),
                "hybrid_prediction": lstm_predictions,
            }
        )
        return fallback_metrics, fallback_frame

    hybrid_predictions = build_weighted_rul_ensemble(
        lstm_predictions=lstm_predictions,
        prophet_predictions=prophet_forecast.predictions,
        config=prophet_config,
    )
    metrics = compute_task_metrics("rul", targets=targets, predictions=hybrid_predictions)
    lstm_metrics = compute_task_metrics("rul", targets=targets, predictions=lstm_predictions)
    prophet_metrics = compute_task_metrics("rul", targets=targets, predictions=prophet_forecast.predictions)
    metrics.update(
        {
            "hybrid_rmse": metrics["rmse"],
            "hybrid_mae": metrics["mae"],
            "hybrid_r2": metrics["r2"],
            "lstm_rmse": lstm_metrics["rmse"],
            "lstm_mae": lstm_metrics["mae"],
            "lstm_r2": lstm_metrics["r2"],
            "prophet_rmse": prophet_metrics["rmse"],
            "prophet_mae": prophet_metrics["mae"],
            "prophet_r2": prophet_metrics["r2"],
            "ensemble_applied": 1.0,
            "prophet_fallback_used": 0.0,
        }
    )
    forecast_frame = pd.DataFrame(
        {
            "lstm_prediction": lstm_predictions,
            "prophet_prediction": prophet_forecast.predictions,
            "prophet_trend": prophet_forecast.trend,
            "prophet_seasonality": prophet_forecast.seasonality,
            "hybrid_prediction": hybrid_predictions,
        }
    )
    return metrics, forecast_frame


def build_weighted_rul_ensemble(
    lstm_predictions: np.ndarray,
    prophet_predictions: np.ndarray,
    config: ProphetEnsembleConfig,
) -> np.ndarray:
    if lstm_predictions.shape != prophet_predictions.shape:
        raise ValueError("lstm_predictions and prophet_predictions must have the same shape.")

    ensemble = (config.lstm_weight * lstm_predictions) + (config.prophet_weight * prophet_predictions)
    return np.asarray(ensemble, dtype=np.float32)


def fit_sequence_model(
    datamodule: CMAPSSSequenceDataModule,
    model_config: SequenceModelConfig,
    training_config: SequenceTrainingConfig,
    output_dir: Path,
    enable_checkpointing: bool = True,
) -> FitResult:
    if not LIGHTNING_AVAILABLE:
        raise RuntimeError(
            "Lightning is not installed in the active environment. Install `lightning` before training the day 9 model."
        )

    seed_everything(training_config.random_state, workers=True)
    datamodule.setup(stage="fit")
    input_dim = len(datamodule.feature_columns_ or [])
    if input_dim == 0:
        raise ValueError("No sequence features were available for model training.")

    model = BidirectionalAttentionSequenceModel(
        input_dim=input_dim,
        target_mode=training_config.target_mode,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        learning_rate=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )

    monitor_name = monitor_name_for_target_mode(training_config.target_mode)
    callbacks: list[Any] = [
        EarlyStopping(
            monitor=monitor_name,
            mode=monitor_mode_for_target_mode(training_config.target_mode),
            patience=training_config.patience,
        )
    ]

    checkpoint_callback = None
    if enable_checkpointing:
        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor=monitor_name,
            mode=monitor_mode_for_target_mode(training_config.target_mode),
            save_top_k=1,
        )
        callbacks.append(checkpoint_callback)

    trainer = Trainer(
        default_root_dir=str(output_dir),
        max_epochs=training_config.max_epochs,
        accelerator=training_config.accelerator,
        devices=1,
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_checkpointing=enable_checkpointing,
    )
    trainer.fit(model, datamodule=datamodule)

    if datamodule.test_df is not None:
        trainer.test(model=model, datamodule=datamodule, verbose=False)

    best_model = model
    checkpoint_path: str | None = None
    if checkpoint_callback is not None and getattr(checkpoint_callback, "best_model_path", ""):
        checkpoint_path = str(checkpoint_callback.best_model_path)
        best_model = BidirectionalAttentionSequenceModel.load_from_checkpoint(checkpoint_path)

    callback_metric = trainer.callback_metrics.get(monitor_name)
    monitor_value = _metric_to_float(callback_metric) if callback_metric is not None else math.nan

    return FitResult(
        model=best_model,
        monitor_name=monitor_name,
        monitor_value=monitor_value,
        checkpoint_path=checkpoint_path,
    )


def predict_from_dataloader(
    model: BidirectionalAttentionSequenceModel,
    dataloader: Any,
    target_mode: TargetMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    attention_rows: list[np.ndarray] = []

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for features, batch_targets in dataloader:
            features = features.to(device)
            outputs, attention_weights = model(features)
            if target_mode == "failure_in_next_window":
                outputs = torch.sigmoid(outputs)

            predictions.append(outputs.detach().cpu().numpy())
            targets.append(batch_targets.detach().cpu().numpy())
            attention_rows.append(attention_weights.detach().cpu().numpy())

    return (
        np.concatenate(predictions).astype(np.float32, copy=False),
        np.concatenate(targets).astype(np.float32, copy=False),
        np.concatenate(attention_rows).astype(np.float32, copy=False),
    )


def compute_task_metrics(
    target_mode: TargetMode,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    if target_mode == "rul":
        rmse = math.sqrt(mean_squared_error(targets, predictions))
        return {
            "rmse": float(rmse),
            "mae": float(mean_absolute_error(targets, predictions)),
            "r2": float(r2_score(targets, predictions)),
        }

    predicted_labels = (predictions >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(targets, predicted_labels)),
        "precision": float(precision_score(targets, predicted_labels, zero_division=0)),
        "recall": float(recall_score(targets, predicted_labels, zero_division=0)),
    }
    if np.unique(targets).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(targets, predictions))
        metrics["pr_auc"] = float(average_precision_score(targets, predictions))
    else:
        metrics["roc_auc"] = math.nan
        metrics["pr_auc"] = math.nan
    return metrics


def build_prediction_frame(
    metadata: pd.DataFrame,
    target_mode: TargetMode,
    target_name: str,
    targets: np.ndarray,
    predictions: np.ndarray,
    attention_weights: np.ndarray,
) -> pd.DataFrame:
    prediction_frame = metadata.reset_index(drop=True).copy()
    prediction_frame[target_name] = targets
    prediction_frame["attention_peak_index"] = attention_weights.argmax(axis=1)
    prediction_frame["attention_peak_weight"] = attention_weights.max(axis=1)

    if target_mode == "rul":
        prediction_frame["prediction"] = predictions
        prediction_frame["absolute_error"] = np.abs(targets - predictions)
    else:
        prediction_frame["predicted_probability"] = predictions
        prediction_frame["predicted_label"] = (predictions >= 0.5).astype(int)

    return prediction_frame


def evaluate_model(
    model: BidirectionalAttentionSequenceModel,
    datamodule: CMAPSSSequenceDataModule,
    target_mode: TargetMode,
    prophet_config: ProphetEnsembleConfig | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    if datamodule.test_dataset is None or datamodule.test_metadata is None or datamodule.target_name_ is None:
        raise ValueError("The datamodule must include test data and be set up before evaluation.")

    lstm_predictions, targets, attention_weights = predict_from_dataloader(
        model=model,
        dataloader=datamodule.test_dataloader(),
        target_mode=target_mode,
    )
    prediction_frame = build_prediction_frame(
        metadata=datamodule.test_metadata,
        target_mode=target_mode,
        target_name=datamodule.target_name_,
        targets=targets,
        predictions=lstm_predictions,
        attention_weights=attention_weights,
    )
    if target_mode == "rul" and prophet_config is not None and prophet_config.enabled:
        if datamodule.train_dataset is None or datamodule.train_metadata is None:
            raise ValueError("The datamodule must expose train windows before fitting the Prophet baseline.")

        train_targets = datamodule.train_dataset.targets.detach().cpu().numpy()
        metrics, prophet_frame = _build_prophet_or_fallback_predictions(
            lstm_predictions=lstm_predictions,
            targets=targets,
            training_metadata=datamodule.train_metadata,
            training_targets=train_targets,
            scoring_metadata=datamodule.test_metadata,
            prophet_config=prophet_config,
        )
        prediction_frame = pd.concat([prediction_frame, prophet_frame], axis=1)
        prediction_frame["prediction"] = prophet_frame["hybrid_prediction"].to_numpy()
        prediction_frame["absolute_error"] = np.abs(targets - prediction_frame["prediction"].to_numpy())
        return metrics, prediction_frame

    metrics = compute_task_metrics(target_mode, targets=targets, predictions=lstm_predictions)
    return metrics, prediction_frame


def optuna_objective(
    trial: Any,
    train_df: pd.DataFrame,
    base_model_config: SequenceModelConfig,
    base_training_config: SequenceTrainingConfig,
    output_dir: Path,
    prophet_config: ProphetEnsembleConfig | None = None,
) -> float:
    trial_params = suggest_hyperparameters(trial, OptunaSearchConfig())
    candidate_model_config, candidate_training_config = apply_trial_parameters(
        model_config=base_model_config,
        training_config=base_training_config,
        trial_params=trial_params,
    )
    datamodule = build_datamodule(
        train_df=train_df,
        test_df=None,
        test_rul_df=None,
        training_config=candidate_training_config,
    )
    fit_result = fit_sequence_model(
        datamodule=datamodule,
        model_config=candidate_model_config,
        training_config=candidate_training_config,
        output_dir=Path(output_dir) / f"trial_{getattr(trial, 'number', 0)}",
        enable_checkpointing=False,
    )
    if (
        prophet_config is not None
        and prophet_config.enabled
        and candidate_training_config.target_mode == "rul"
        and datamodule.val_dataset is not None
        and datamodule.val_metadata is not None
        and datamodule.train_dataset is not None
        and datamodule.train_metadata is not None
    ):
        val_predictions, val_targets, _ = predict_from_dataloader(
            model=fit_result.model,
            dataloader=datamodule.val_dataloader(),
            target_mode=candidate_training_config.target_mode,
        )
        metrics, _ = _build_prophet_or_fallback_predictions(
            lstm_predictions=val_predictions,
            targets=val_targets,
            training_metadata=datamodule.train_metadata,
            training_targets=datamodule.train_dataset.targets.detach().cpu().numpy(),
            scoring_metadata=datamodule.val_metadata,
            prophet_config=prophet_config,
        )
        return metrics["rmse"]
    return fit_result.monitor_value


def run_optuna_search(
    train_df: pd.DataFrame,
    base_model_config: SequenceModelConfig,
    base_training_config: SequenceTrainingConfig,
    search_config: OptunaSearchConfig,
    output_dir: Path,
    prophet_config: ProphetEnsembleConfig | None = None,
) -> HyperparameterSearchResult:
    if search_config.n_trials < 1:
        return HyperparameterSearchResult(
            best_model_config=base_model_config,
            best_training_config=base_training_config,
            best_value=math.nan,
            trials_frame=pd.DataFrame(),
        )

    optuna = _load_optuna_runtime()
    sampler = optuna.samplers.TPESampler(seed=base_training_config.random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(search_config.n_trials, 3))
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: _optuna_objective_with_search_config(
            trial=trial,
            train_df=train_df,
            base_model_config=base_model_config,
            base_training_config=base_training_config,
            search_config=search_config,
            output_dir=output_dir,
            prophet_config=prophet_config,
        ),
        n_trials=search_config.n_trials,
    )

    best_model_config, best_training_config = apply_trial_parameters(
        model_config=base_model_config,
        training_config=base_training_config,
        trial_params=study.best_trial.params,
    )
    return HyperparameterSearchResult(
        best_model_config=best_model_config,
        best_training_config=best_training_config,
        best_value=float(study.best_value),
        trials_frame=study.trials_dataframe(),
    )


def _optuna_objective_with_search_config(
    trial: Any,
    train_df: pd.DataFrame,
    base_model_config: SequenceModelConfig,
    base_training_config: SequenceTrainingConfig,
    search_config: OptunaSearchConfig,
    output_dir: Path,
    prophet_config: ProphetEnsembleConfig | None = None,
) -> float:
    trial_params = suggest_hyperparameters(trial, search_config)
    candidate_model_config, candidate_training_config = apply_trial_parameters(
        model_config=base_model_config,
        training_config=base_training_config,
        trial_params=trial_params,
    )
    datamodule = build_datamodule(
        train_df=train_df,
        test_df=None,
        test_rul_df=None,
        training_config=candidate_training_config,
    )
    fit_result = fit_sequence_model(
        datamodule=datamodule,
        model_config=candidate_model_config,
        training_config=candidate_training_config,
        output_dir=Path(output_dir) / f"trial_{getattr(trial, 'number', 0)}",
        enable_checkpointing=False,
    )
    if (
        prophet_config is not None
        and prophet_config.enabled
        and candidate_training_config.target_mode == "rul"
        and datamodule.val_dataset is not None
        and datamodule.val_metadata is not None
        and datamodule.train_dataset is not None
        and datamodule.train_metadata is not None
    ):
        val_predictions, val_targets, _ = predict_from_dataloader(
            model=fit_result.model,
            dataloader=datamodule.val_dataloader(),
            target_mode=candidate_training_config.target_mode,
        )
        metrics, _ = _build_prophet_or_fallback_predictions(
            lstm_predictions=val_predictions,
            targets=val_targets,
            training_metadata=datamodule.train_metadata,
            training_targets=datamodule.train_dataset.targets.detach().cpu().numpy(),
            scoring_metadata=datamodule.val_metadata,
            prophet_config=prophet_config,
        )
        return metrics["rmse"]
    return fit_result.monitor_value


def save_sequence_artifacts(
    output_root: Path,
    run_name: str,
    model_config: SequenceModelConfig,
    training_config: SequenceTrainingConfig,
    prophet_config: ProphetEnsembleConfig | None,
    metrics: dict[str, float],
    predictions: pd.DataFrame,
    search_result: HyperparameterSearchResult | None,
    checkpoint_path: str | None,
) -> SavedSequenceArtifacts:
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = output_dir / "metrics.json"
    predictions_csv = output_dir / "test_predictions.csv"
    metadata_json = output_dir / "run_metadata.json"
    summary_markdown = output_dir / "day_09_summary.md"

    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(predictions_csv, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
    }
    if prophet_config is not None and prophet_config.enabled and training_config.target_mode == "rul":
        metadata["prophet_ensemble_config"] = asdict(prophet_config)
    if search_result is not None and not search_result.trials_frame.empty:
        metadata["optuna_best_value"] = search_result.best_value
        metadata["optuna_trial_count"] = int(len(search_result.trials_frame))
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_lines = [
        "# Day 09 Sequence Model Summary",
        "",
        f"- Target mode: `{training_config.target_mode}`",
        f"- Window size: `{training_config.window_size}`",
        f"- Hidden size: `{model_config.hidden_size}`",
        f"- Bidirectional LSTM layers: `{model_config.num_layers}`",
        f"- Dropout: `{model_config.dropout:.3f}`",
        f"- Learning rate: `{model_config.learning_rate:.6f}`",
    ]
    if prophet_config is not None and prophet_config.enabled and training_config.target_mode == "rul":
        summary_lines.extend(
            [
                f"- Prophet ensemble enabled: `{prophet_config.enabled}`",
                f"- Hybrid weights: `LSTM={prophet_config.lstm_weight:.1f}`, `Prophet={prophet_config.prophet_weight:.1f}`",
                f"- Prophet seasonality period: `{prophet_config.seasonality_period:.1f}`",
            ]
        )
    summary_lines.extend(
        [
            "",
            "## Test Metrics",
            "",
        ]
    )
    for metric_name, metric_value in metrics.items():
        summary_lines.append(f"- {metric_name}: `{metric_value:.6f}`")

    trials_csv: Path | None = None
    if search_result is not None and not search_result.trials_frame.empty:
        trials_csv = output_dir / "optuna_trials.csv"
        search_result.trials_frame.to_csv(trials_csv, index=False)
        summary_lines.extend(
            [
                "",
                "## Optuna Search",
                "",
                f"- Trials: `{len(search_result.trials_frame)}`",
                f"- Best objective value: `{search_result.best_value:.6f}`",
            ]
        )

    summary_markdown.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return SavedSequenceArtifacts(
        output_dir=output_dir,
        metrics_json=metrics_json,
        predictions_csv=predictions_csv,
        metadata_json=metadata_json,
        summary_markdown=summary_markdown,
        trials_csv=trials_csv,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
    )


def train_day9_sequence_model(
    fd: str = "FD001",
    model_config: SequenceModelConfig | None = None,
    training_config: SequenceTrainingConfig | None = None,
    search_config: OptunaSearchConfig | None = None,
    prophet_config: ProphetEnsembleConfig | None = None,
    run_name: str = "",
    output_dir: str = "Data/experiments/day9_sequence_training",
) -> tuple[dict[str, float], pd.DataFrame, SavedSequenceArtifacts]:
    model_config = model_config or SequenceModelConfig()
    training_config = training_config or SequenceTrainingConfig()
    search_config = search_config or OptunaSearchConfig()
    prophet_config = prophet_config or ProphetEnsembleConfig()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "Data"
    train_df, test_df, test_rul_df = load_dataset(data_dir=data_dir, source="nasa", fd=fd)

    best_model_config = model_config
    best_training_config = training_config
    search_result: HyperparameterSearchResult | None = None
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    resolved_run_name = run_name.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / resolved_run_name

    if search_config.n_trials > 0:
        search_result = run_optuna_search(
            train_df=train_df,
            base_model_config=model_config,
            base_training_config=training_config,
            search_config=search_config,
            output_dir=run_root / "optuna",
            prophet_config=prophet_config,
        )
        best_model_config = search_result.best_model_config
        best_training_config = search_result.best_training_config

    datamodule = build_datamodule(
        train_df=train_df,
        test_df=test_df,
        test_rul_df=test_rul_df,
        training_config=best_training_config,
    )
    fit_result = fit_sequence_model(
        datamodule=datamodule,
        model_config=best_model_config,
        training_config=best_training_config,
        output_dir=run_root / "final_fit",
        enable_checkpointing=True,
    )
    metrics, predictions = evaluate_model(
        model=fit_result.model,
        datamodule=datamodule,
        target_mode=best_training_config.target_mode,
        prophet_config=prophet_config,
    )

    artifacts = save_sequence_artifacts(
        output_root=output_root,
        run_name=resolved_run_name,
        model_config=best_model_config,
        training_config=best_training_config,
        prophet_config=prophet_config,
        metrics=metrics,
        predictions=predictions,
        search_result=search_result,
        checkpoint_path=fit_result.checkpoint_path,
    )
    return metrics, predictions, artifacts

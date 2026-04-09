from __future__ import annotations

import json
import math
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
        if num_layers < 2 or num_layers > 3:
            raise ValueError("num_layers must be either 2 or 3 for the day 9 architecture.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

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
) -> tuple[dict[str, float], pd.DataFrame]:
    if datamodule.test_dataset is None or datamodule.test_metadata is None or datamodule.target_name_ is None:
        raise ValueError("The datamodule must include test data and be set up before evaluation.")

    predictions, targets, attention_weights = predict_from_dataloader(
        model=model,
        dataloader=datamodule.test_dataloader(),
        target_mode=target_mode,
    )
    metrics = compute_task_metrics(target_mode, targets=targets, predictions=predictions)
    prediction_frame = build_prediction_frame(
        metadata=datamodule.test_metadata,
        target_mode=target_mode,
        target_name=datamodule.target_name_,
        targets=targets,
        predictions=predictions,
        attention_weights=attention_weights,
    )
    return metrics, prediction_frame


def optuna_objective(
    trial: Any,
    train_df: pd.DataFrame,
    base_model_config: SequenceModelConfig,
    base_training_config: SequenceTrainingConfig,
    output_dir: Path,
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
    return fit_result.monitor_value


def run_optuna_search(
    train_df: pd.DataFrame,
    base_model_config: SequenceModelConfig,
    base_training_config: SequenceTrainingConfig,
    search_config: OptunaSearchConfig,
    output_dir: Path,
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
    return fit_result.monitor_value


def save_sequence_artifacts(
    output_root: Path,
    run_name: str,
    model_config: SequenceModelConfig,
    training_config: SequenceTrainingConfig,
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
        "",
        "## Test Metrics",
        "",
    ]
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
    run_name: str = "",
    output_dir: str = "Data/experiments/day9_sequence_training",
) -> tuple[dict[str, float], pd.DataFrame, SavedSequenceArtifacts]:
    model_config = model_config or SequenceModelConfig()
    training_config = training_config or SequenceTrainingConfig()
    search_config = search_config or OptunaSearchConfig()

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
    )

    artifacts = save_sequence_artifacts(
        output_root=output_root,
        run_name=resolved_run_name,
        model_config=best_model_config,
        training_config=best_training_config,
        metrics=metrics,
        predictions=predictions,
        search_result=search_result,
        checkpoint_path=fit_result.checkpoint_path,
    )
    return metrics, predictions, artifacts

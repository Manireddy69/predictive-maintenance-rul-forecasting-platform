from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .anomaly_baseline import (
    ANOMALY_FEATURE_COLUMNS,
    AnomalyEvaluation,
    build_holdout_with_injected_anomalies,
    evaluate_anomaly_scores,
    evaluation_summary_frame,
    score_anomaly_detectors,
)
from .data import load_dataset
from .mlflow_tracking import log_week1_anomaly_checkpoint


@dataclass(frozen=True)
class SequenceWindowSet:
    windows: np.ndarray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class LSTMAutoencoderConfig:
    sequence_length: int = 30
    stride: int = 1
    hidden_dim: int = 64
    latent_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 20
    contamination: float = 0.08
    device: str | None = None


@dataclass(frozen=True)
class SavedExperimentArtifacts:
    output_dir: Path
    comparison_csv: Path
    window_scores_csv: Path
    row_scores_csv: Path
    training_history_csv: Path
    metadata_json: Path
    summary_markdown: Path
    mlflow_summary_json: Path | None = None


def _load_torch_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "PyTorch is not installed. Add `torch` to your environment before training the LSTM autoencoder."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def torch_is_available() -> bool:
    try:
        _load_torch_runtime()
    except RuntimeError:
        return False
    return True


def build_sequence_windows(
    df: pd.DataFrame,
    sequence_length: int = 30,
    stride: int = 1,
    feature_columns: list[str] | None = None,
    group_column: str = "unit",
    sort_column: str = "cycle",
    label_column: str | None = None,
) -> SequenceWindowSet:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    feature_columns = feature_columns or ANOMALY_FEATURE_COLUMNS
    required_columns = {group_column, sort_column, *feature_columns}
    if label_column is not None:
        required_columns.add(label_column)

    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Cannot build sequence windows without columns: {missing_str}")

    sorted_df = df.sort_values([group_column, sort_column]).reset_index(drop=True)
    windows: list[np.ndarray] = []
    metadata_rows: list[dict[str, int | float | str]] = []

    for group_value, group_df in sorted_df.groupby(group_column, sort=False):
        feature_values = group_df.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
        sort_values = group_df.loc[:, sort_column].to_numpy(copy=False)
        labels = None
        if label_column is not None:
            labels = group_df.loc[:, label_column].to_numpy(dtype=int, copy=False)

        if len(group_df) < sequence_length:
            continue

        for start_idx in range(0, len(group_df) - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            windows.append(feature_values[start_idx:end_idx])

            metadata = {
                group_column: group_value,
                "window_start_index": int(start_idx),
                "window_end_index": int(end_idx - 1),
                "window_length": int(sequence_length),
                f"{sort_column}_start": int(sort_values[start_idx]),
                f"{sort_column}_end": int(sort_values[end_idx - 1]),
            }
            if labels is not None:
                metadata["is_anomaly_window"] = int(labels[start_idx:end_idx].max())
            metadata_rows.append(metadata)

    if windows:
        window_array = np.stack(windows).astype(float, copy=False)
    else:
        window_array = np.empty((0, sequence_length, len(feature_columns)), dtype=float)

    return SequenceWindowSet(windows=window_array, metadata=pd.DataFrame(metadata_rows))


def fit_sequence_scaler(train_windows: np.ndarray) -> StandardScaler:
    if train_windows.ndim != 3:
        raise ValueError("train_windows must be a 3D array of shape (num_windows, sequence_length, num_features).")
    if train_windows.shape[0] == 0:
        raise ValueError("train_windows must contain at least one window.")

    scaler = StandardScaler()
    scaler.fit(train_windows.reshape(-1, train_windows.shape[-1]))
    return scaler


def transform_sequence_windows(windows: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    if windows.ndim != 3:
        raise ValueError("windows must be a 3D array of shape (num_windows, sequence_length, num_features).")
    if windows.shape[0] == 0:
        return windows.astype(float, copy=True)

    transformed = scaler.transform(windows.reshape(-1, windows.shape[-1]))
    return transformed.reshape(windows.shape).astype(float, copy=False)


def aggregate_window_scores(
    scored_df: pd.DataFrame,
    score_column: str,
    sequence_length: int,
    stride: int = 1,
    group_column: str = "unit",
    sort_column: str = "cycle",
    label_column: str = "is_anomaly",
) -> pd.DataFrame:
    window_set = build_sequence_windows(
        scored_df,
        sequence_length=sequence_length,
        stride=stride,
        feature_columns=[score_column],
        group_column=group_column,
        sort_column=sort_column,
        label_column=label_column,
    )

    aggregated = window_set.metadata.copy()
    if window_set.windows.shape[0] == 0:
        aggregated[score_column] = pd.Series(dtype=float)
        return aggregated

    aggregated[score_column] = window_set.windows.max(axis=(1, 2))
    return aggregated


def _build_lstm_autoencoder_class(nn: Any) -> type[Any]:
    class LSTMAutoencoder(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            num_layers: int = 1,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.to_latent = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.LSTM(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.output_layer = nn.Linear(hidden_dim, input_dim)

        def forward(self, inputs: Any) -> Any:
            _, (hidden_state, _) = self.encoder(inputs)
            latent = self.to_latent(hidden_state[-1])
            decoder_inputs = latent.unsqueeze(1).repeat(1, inputs.size(1), 1)
            decoded, _ = self.decoder(decoder_inputs)
            return self.output_layer(decoded)

    return LSTMAutoencoder


def train_lstm_autoencoder(
    train_windows: np.ndarray,
    config: LSTMAutoencoderConfig,
) -> tuple[Any, pd.DataFrame, str]:
    torch, nn, DataLoader, TensorDataset = _load_torch_runtime()
    model_class = _build_lstm_autoencoder_class(nn)

    device_name = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    model = model_class(
        input_dim=train_windows.shape[-1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    dataset = TensorDataset(torch.tensor(train_windows, dtype=torch.float32))
    batch_size = min(config.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    history_rows: list[dict[str, float | int]] = []

    model.train()
    for epoch in range(1, config.num_epochs + 1):
        cumulative_loss = 0.0
        sample_count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()

            batch_size_actual = int(batch.size(0))
            cumulative_loss += float(loss.item()) * batch_size_actual
            sample_count += batch_size_actual

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": cumulative_loss / max(sample_count, 1),
            }
        )

    return model, pd.DataFrame(history_rows), str(device)


def compute_reconstruction_errors(
    model: Any,
    windows: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    torch, _, DataLoader, TensorDataset = _load_torch_runtime()
    if windows.shape[0] == 0:
        return np.empty(0, dtype=float)

    dataset = TensorDataset(torch.tensor(windows, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    device = next(model.parameters()).device

    errors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            batch_errors = torch.mean((reconstructed - batch) ** 2, dim=(1, 2))
            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors).astype(float, copy=False)


def reconstruction_error_threshold(
    train_errors: np.ndarray,
    contamination: float = 0.08,
) -> float:
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")
    if train_errors.size == 0:
        raise ValueError("train_errors must contain at least one value.")

    return float(np.quantile(train_errors, 1.0 - contamination))


def run_lstm_autoencoder_experiment(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    config: LSTMAutoencoderConfig | None = None,
    anomaly_fraction: float = 0.08,
    random_state: int = 42,
) -> tuple[pd.DataFrame, AnomalyEvaluation, pd.DataFrame, float, str]:
    config = config or LSTMAutoencoderConfig()

    temporal_holdout_df = build_holdout_with_injected_anomalies(
        holdout_df,
        anomaly_fraction=anomaly_fraction,
        random_state=random_state,
        append_anomalies=False,
    )

    train_window_set = build_sequence_windows(
        train_df,
        sequence_length=config.sequence_length,
        stride=config.stride,
    )
    eval_window_set = build_sequence_windows(
        temporal_holdout_df,
        sequence_length=config.sequence_length,
        stride=config.stride,
        label_column="is_anomaly",
    )

    if train_window_set.windows.shape[0] == 0:
        raise ValueError("Training data did not produce any sequence windows for the LSTM autoencoder.")
    if eval_window_set.windows.shape[0] == 0:
        raise ValueError("Evaluation data did not produce any sequence windows for the LSTM autoencoder.")

    scaler = fit_sequence_scaler(train_window_set.windows)
    train_windows_scaled = transform_sequence_windows(train_window_set.windows, scaler)
    eval_windows_scaled = transform_sequence_windows(eval_window_set.windows, scaler)

    model, history_df, device_name = train_lstm_autoencoder(train_windows_scaled, config=config)
    train_errors = compute_reconstruction_errors(model, train_windows_scaled, batch_size=config.batch_size)
    eval_errors = compute_reconstruction_errors(model, eval_windows_scaled, batch_size=config.batch_size)
    threshold = reconstruction_error_threshold(train_errors, contamination=config.contamination)

    result = evaluate_anomaly_scores(
        eval_window_set.metadata["is_anomaly_window"].to_numpy(dtype=int, copy=False),
        eval_errors,
        model_name="LSTM Autoencoder",
    )

    evaluation_df = eval_window_set.metadata.copy()
    evaluation_df["reconstruction_error"] = eval_errors
    evaluation_df["predicted_anomaly"] = (eval_errors >= threshold).astype(int)
    return evaluation_df, result, history_df, threshold, device_name


def run_window_level_comparison_experiment(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    config: LSTMAutoencoderConfig | None = None,
    anomaly_fraction: float = 0.08,
    random_state: int = 42,
    contamination: float = 0.08,
    n_neighbors: int = 35,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, str]:
    config = config or LSTMAutoencoderConfig(contamination=contamination)

    temporal_holdout_df = build_holdout_with_injected_anomalies(
        holdout_df,
        anomaly_fraction=anomaly_fraction,
        random_state=random_state,
        append_anomalies=False,
    )
    scored_holdout_df = score_anomaly_detectors(
        train_df=train_df,
        evaluation_df=temporal_holdout_df,
        contamination=contamination,
        random_state=random_state,
        n_neighbors=n_neighbors,
    )

    lstm_eval_df, lstm_result, history_df, threshold, device_name = run_lstm_autoencoder_experiment(
        train_df=train_df,
        holdout_df=holdout_df,
        config=config,
        anomaly_fraction=anomaly_fraction,
        random_state=random_state,
    )

    combined_window_scores = lstm_eval_df.copy()
    results: dict[str, AnomalyEvaluation] = {"lstm_autoencoder": lstm_result}

    for score_column, model_key, model_name in (
        ("isolation_forest_score", "isolation_forest", "Isolation Forest"),
        ("local_outlier_factor_score", "local_outlier_factor", "Local Outlier Factor"),
        ("zscore_score", "zscore", "Z-Score Distance"),
        ("mad_score", "mad", "MAD Distance"),
    ):
        score_windows = aggregate_window_scores(
            scored_holdout_df,
            score_column=score_column,
            sequence_length=config.sequence_length,
            stride=config.stride,
        )
        combined_window_scores = combined_window_scores.merge(
            score_windows,
            on=[
                "unit",
                "window_start_index",
                "window_end_index",
                "window_length",
                "cycle_start",
                "cycle_end",
                "is_anomaly_window",
            ],
            how="left",
        )
        results[model_key] = evaluate_anomaly_scores(
            combined_window_scores["is_anomaly_window"].to_numpy(dtype=int, copy=False),
            combined_window_scores[score_column].to_numpy(dtype=float, copy=False),
            model_name=model_name,
        )

    comparison_df = evaluation_summary_frame(results)
    return comparison_df, combined_window_scores, scored_holdout_df, history_df, threshold, device_name


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_comparison_artifacts(
    comparison_df: pd.DataFrame,
    combined_window_scores: pd.DataFrame,
    scored_holdout_df: pd.DataFrame,
    history_df: pd.DataFrame,
    threshold: float,
    device_name: str,
    output_root: Path,
    run_name: str,
    config: LSTMAutoencoderConfig,
    anomaly_fraction: float,
    random_state: int,
) -> SavedExperimentArtifacts:
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_csv = output_dir / "comparison_metrics.csv"
    window_scores_csv = output_dir / "window_scores.csv"
    row_scores_csv = output_dir / "row_level_scores.csv"
    training_history_csv = output_dir / "training_history.csv"
    metadata_json = output_dir / "run_metadata.json"
    summary_markdown = output_dir / "day_06_summary.md"

    comparison_df.to_csv(comparison_csv, index=False)
    combined_window_scores.to_csv(window_scores_csv, index=False)
    scored_holdout_df.to_csv(row_scores_csv, index=False)
    history_df.to_csv(training_history_csv, index=False)

    best_row = comparison_df.iloc[0]
    metadata = {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "device_name": device_name,
        "anomaly_fraction": anomaly_fraction,
        "random_state": random_state,
        "config": {
            "sequence_length": config.sequence_length,
            "stride": config.stride,
            "hidden_dim": config.hidden_dim,
            "latent_dim": config.latent_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "contamination": config.contamination,
            "device": config.device,
        },
        "best_model": {
            "model_key": str(best_row["model_key"]),
            "model_name": str(best_row["model_name"]),
            "roc_auc": float(best_row["roc_auc"]),
            "pr_auc": float(best_row["pr_auc"]),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_df = comparison_df.copy()
    summary_df["roc_auc"] = summary_df["roc_auc"].map(lambda value: f"{float(value):.6f}")
    summary_df["pr_auc"] = summary_df["pr_auc"].map(lambda value: f"{float(value):.6f}")

    summary_lines = [
        "# Day 06 Anomaly Detection Summary",
        "",
        f"- Run name: `{run_name}`",
        f"- Best model: `{best_row['model_name']}`",
        f"- Best PR-AUC: `{float(best_row['pr_auc']):.6f}`",
        f"- Reconstruction-error threshold: `{threshold:.6f}`",
        f"- Training device: `{device_name}`",
        f"- Final training loss: `{float(history_df['train_loss'].iloc[-1]):.6f}`",
        "",
        "## Comparison Table",
        "",
        _markdown_table(summary_df),
        "",
        "## Plain-Language Takeaway",
        "",
        (
            "The LSTM autoencoder learned useful temporal structure, but the simplest robust statistical "
            f"method in this run was `{best_row['model_name']}`."
        ),
        (
            "That means deep learning is competitive here, but not automatically better than "
            "well-chosen statistical baselines."
        ),
    ]
    summary_markdown.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return SavedExperimentArtifacts(
        output_dir=output_dir,
        comparison_csv=comparison_csv,
        window_scores_csv=window_scores_csv,
        row_scores_csv=row_scores_csv,
        training_history_csv=training_history_csv,
        metadata_json=metadata_json,
        summary_markdown=summary_markdown,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM autoencoder and compare it against anomaly baselines.")
    parser.add_argument("--synthetic-units", type=int, default=40, help="Total synthetic units across train and hold-out.")
    parser.add_argument(
        "--synthetic-test-fraction",
        type=float,
        default=0.25,
        help="Fraction of units reserved for the hold-out split.",
    )
    parser.add_argument(
        "--anomaly-fraction",
        type=float,
        default=0.08,
        help="Fraction of hold-out rows perturbed into anomalies.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.08,
        help="Expected anomaly rate used for thresholding and detector configuration.",
    )
    parser.add_argument("--sequence-length", type=int, default=30, help="Sliding window size for the LSTM autoencoder.")
    parser.add_argument("--stride", type=int, default=5, help="Sliding window step size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for autoencoder training.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for autoencoder training.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for the encoder and decoder LSTMs.")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent bottleneck dimension for the autoencoder.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Data/experiments/anomaly_day6",
        help="Directory where experiment artifacts will be saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional folder name for the saved run. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log the saved anomaly comparison run to MLflow.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="week1-anomaly-checkpoint",
        help="MLflow experiment name used when --log-mlflow is enabled.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="",
        help="Optional MLflow tracking URI. Defaults to a local file-based mlruns directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "Data"
    train_df, holdout_df, _ = load_dataset(
        data_dir=data_dir,
        source="synthetic",
        synthetic_units=args.synthetic_units,
        synthetic_test_fraction=args.synthetic_test_fraction,
        synthetic_anomaly_fraction=0.0,
        synthetic_random_state=args.random_state,
    )

    config = LSTMAutoencoderConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        contamination=args.contamination,
    )

    comparison_df, combined_window_scores, scored_holdout_df, history_df, threshold, device_name = run_window_level_comparison_experiment(
        train_df=train_df,
        holdout_df=holdout_df,
        config=config,
        anomaly_fraction=args.anomaly_fraction,
        random_state=args.random_state,
        contamination=args.contamination,
    )

    run_name = args.run_name.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    artifacts = save_comparison_artifacts(
        comparison_df=comparison_df,
        combined_window_scores=combined_window_scores,
        scored_holdout_df=scored_holdout_df,
        history_df=history_df,
        threshold=threshold,
        device_name=device_name,
        output_root=output_root,
        run_name=run_name,
        config=config,
        anomaly_fraction=args.anomaly_fraction,
        random_state=args.random_state,
    )

    mlflow_run_id: str | None = None
    if args.log_mlflow:
        metadata = json.loads(artifacts.metadata_json.read_text(encoding="utf-8"))
        mlflow_run_id = log_week1_anomaly_checkpoint(
            comparison_df=comparison_df,
            metadata=metadata,
            artifact_paths=[
                artifacts.comparison_csv,
                artifacts.window_scores_csv,
                artifacts.row_scores_csv,
                artifacts.training_history_csv,
                artifacts.metadata_json,
                artifacts.summary_markdown,
            ],
            project_root=project_root,
            experiment_name=args.mlflow_experiment,
            run_name=run_name,
            tracking_uri=args.mlflow_tracking_uri.strip() or None,
        )

    print("=== Window-level anomaly detection comparison ===")
    print(comparison_df.to_string(index=False))
    print()
    print(f"Reconstruction-error threshold: {threshold:.6f}")
    print(f"Training device: {device_name}")
    print(f"Final training loss: {history_df['train_loss'].iloc[-1]:.6f}")
    print(f"Saved artifacts: {artifacts.output_dir}")
    print(f"Summary report: {artifacts.summary_markdown}")
    if mlflow_run_id is not None:
        print(f"MLflow run id: {mlflow_run_id}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .data import OPERATIONAL_SETTING_COLUMNS, add_train_rul, get_sensor_columns

try:  # pragma: no cover - depends on installed package variant
    from lightning.pytorch import LightningDataModule
except ImportError:  # pragma: no cover - depends on installed package variant
    try:
        from pytorch_lightning import LightningDataModule
    except ImportError:  # pragma: no cover - lightweight fallback for local tests
        class LightningDataModule:  # type: ignore[no-redef]
            """Fallback base class when Lightning is not installed."""


TargetMode = Literal["rul", "failure_in_next_window"]


@dataclass(frozen=True)
class SequenceWindowDataset:
    features: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame
    feature_columns: list[str]
    target_name: str


class SequenceTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have the same number of samples.")

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def default_sequence_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [column for column in OPERATIONAL_SETTING_COLUMNS if column in df.columns]
    feature_columns.extend(get_sensor_columns(df))
    if not feature_columns:
        raise ValueError("No sequence feature columns were found in the input dataframe.")
    return feature_columns


def add_test_rul(
    df: pd.DataFrame,
    rul_df: pd.DataFrame,
    group_column: str = "unit",
    sort_column: str = "cycle",
    rul_column: str = "rul",
) -> pd.DataFrame:
    required_df_columns = {group_column, sort_column}
    missing_df = required_df_columns.difference(df.columns)
    if missing_df:
        missing_str = ", ".join(sorted(missing_df))
        raise ValueError(f"Cannot attach test RUL without columns: {missing_str}")

    if rul_column not in rul_df.columns:
        raise ValueError(f"Cannot attach test RUL without `{rul_column}` in the RUL dataframe.")

    last_cycles = df.groupby(group_column)[sort_column].max().rename("observed_last_cycle")
    unit_rul = rul_df.loc[:, [rul_column]].rename_axis(group_column)
    merged = last_cycles.to_frame().join(unit_rul, how="left")
    if merged[rul_column].isna().any():
        missing_units = merged.index[merged[rul_column].isna()].tolist()
        raise ValueError(f"Missing RUL targets for units: {missing_units}")

    total_cycles = (merged["observed_last_cycle"] + merged[rul_column]).rename("total_cycles_to_failure")
    result = df.copy()
    result = result.merge(total_cycles, left_on=group_column, right_index=True, how="left")
    result[rul_column] = result["total_cycles_to_failure"] - result[sort_column]
    result.drop(columns=["total_cycles_to_failure"], inplace=True)
    return result


def build_target_sequences(
    df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 1,
    feature_columns: list[str] | None = None,
    target_mode: TargetMode = "rul",
    prediction_horizon: int | None = None,
    group_column: str = "unit",
    sort_column: str = "cycle",
    rul_column: str = "rul",
) -> SequenceWindowDataset:
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if target_mode not in {"rul", "failure_in_next_window"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    feature_columns = feature_columns or default_sequence_feature_columns(df)
    required_columns = {group_column, sort_column, rul_column, *feature_columns}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Cannot build target sequences without columns: {missing_str}")

    effective_horizon = prediction_horizon or window_size
    if effective_horizon < 1:
        raise ValueError("prediction_horizon must be at least 1.")

    sorted_df = df.sort_values([group_column, sort_column]).reset_index(drop=True)
    window_features: list[np.ndarray] = []
    window_targets: list[float] = []
    metadata_rows: list[dict[str, int | float | str]] = []

    for group_value, group_df in sorted_df.groupby(group_column, sort=False):
        if len(group_df) < window_size:
            continue

        feature_values = group_df.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=False)
        cycle_values = group_df.loc[:, sort_column].to_numpy(dtype=int, copy=False)
        rul_values = group_df.loc[:, rul_column].to_numpy(dtype=np.float32, copy=False)

        for start_idx in range(0, len(group_df) - window_size + 1, stride):
            end_idx = start_idx + window_size
            target_idx = end_idx - 1
            target_rul = float(rul_values[target_idx])

            if target_mode == "rul":
                target_value = target_rul
                target_name = rul_column
            else:
                target_value = float(1 <= target_rul <= effective_horizon)
                target_name = "failure_in_next_window"

            window_features.append(feature_values[start_idx:end_idx])
            metadata_rows.append(
                {
                    group_column: group_value,
                    "window_start_index": int(start_idx),
                    "window_end_index": int(end_idx - 1),
                    "window_size": int(window_size),
                    f"{sort_column}_start": int(cycle_values[start_idx]),
                    f"{sort_column}_end": int(cycle_values[end_idx - 1]),
                    "target_index": int(target_idx),
                    "target_cycle": int(cycle_values[target_idx]),
                    "target_rul": target_rul,
                    "prediction_horizon": int(effective_horizon),
                    target_name: target_value,
                }
            )
            window_targets.append(target_value)

    if window_features:
        features = np.stack(window_features).astype(np.float32, copy=False)
        targets = np.asarray(window_targets, dtype=np.float32)
        metadata = pd.DataFrame(metadata_rows)
    else:
        features = np.empty((0, window_size, len(feature_columns)), dtype=np.float32)
        targets = np.empty(0, dtype=np.float32)
        metadata = pd.DataFrame(
            columns=[
                group_column,
                "window_start_index",
                "window_end_index",
                "window_size",
                f"{sort_column}_start",
                f"{sort_column}_end",
                "target_index",
                "target_cycle",
                "target_rul",
                "prediction_horizon",
                "failure_in_next_window" if target_mode == "failure_in_next_window" else rul_column,
            ]
        )
        target_name = "failure_in_next_window" if target_mode == "failure_in_next_window" else rul_column

    return SequenceWindowDataset(
        features=features,
        targets=targets,
        metadata=metadata,
        feature_columns=list(feature_columns),
        target_name=target_name,
    )


def fit_sequence_scaler(train_features: np.ndarray) -> StandardScaler:
    if train_features.ndim != 3:
        raise ValueError("train_features must have shape (num_samples, window_size, num_features).")
    if train_features.shape[0] == 0:
        raise ValueError("train_features must contain at least one sample.")

    scaler = StandardScaler()
    scaler.fit(train_features.reshape(-1, train_features.shape[-1]))
    return scaler


def transform_sequence_features(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError("features must have shape (num_samples, window_size, num_features).")
    if features.shape[0] == 0:
        return features.astype(np.float32, copy=True)

    transformed = scaler.transform(features.reshape(-1, features.shape[-1]))
    return transformed.reshape(features.shape).astype(np.float32, copy=False)


def split_train_validation_by_unit(
    df: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    group_column: str = "unit",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in the range [0, 1).")
    if group_column not in df.columns:
        raise ValueError(f"Cannot split by `{group_column}` because it is missing from the dataframe.")

    unique_units = np.asarray(sorted(df[group_column].unique()))
    if unique_units.size < 2 and validation_fraction > 0:
        raise ValueError("At least two units are required to create a train/validation split.")

    validation_count = int(round(unique_units.size * validation_fraction))
    validation_count = min(max(validation_count, 1 if validation_fraction > 0 else 0), max(unique_units.size - 1, 0))

    rng = np.random.default_rng(random_state)
    shuffled_units = unique_units.copy()
    rng.shuffle(shuffled_units)

    validation_units = set(shuffled_units[:validation_count].tolist())
    train_mask = ~df[group_column].isin(validation_units)
    validation_mask = df[group_column].isin(validation_units)
    return df.loc[train_mask].copy(), df.loc[validation_mask].copy()


class CMAPSSSequenceDataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        test_rul_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
        feature_columns: list[str] | None = None,
        target_mode: TargetMode = "rul",
        window_size: int = 30,
        stride: int = 1,
        prediction_horizon: int | None = None,
        validation_fraction: float = 0.2,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.train_df = train_df.copy()
        self.test_df = test_df.copy() if test_df is not None else None
        self.test_rul_df = test_rul_df.copy() if test_rul_df is not None else None
        self.val_df = val_df.copy() if val_df is not None else None
        self.feature_columns = feature_columns
        self.target_mode = target_mode
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.validation_fraction = validation_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.random_state = random_state

        self.scaler: StandardScaler | None = None
        self.train_dataset: SequenceTensorDataset | None = None
        self.val_dataset: SequenceTensorDataset | None = None
        self.test_dataset: SequenceTensorDataset | None = None
        self.train_metadata: pd.DataFrame | None = None
        self.val_metadata: pd.DataFrame | None = None
        self.test_metadata: pd.DataFrame | None = None
        self.feature_columns_: list[str] | None = None
        self.target_name_: str | None = None

    def setup(self, stage: str | None = None) -> None:
        train_source = add_train_rul(self.train_df) if "rul" not in self.train_df.columns else self.train_df.copy()

        if self.val_df is None:
            train_source, val_source = split_train_validation_by_unit(
                train_source,
                validation_fraction=self.validation_fraction,
                random_state=self.random_state,
            )
        else:
            val_source = add_train_rul(self.val_df) if "rul" not in self.val_df.columns else self.val_df.copy()

        self.feature_columns_ = self.feature_columns or default_sequence_feature_columns(train_source)

        train_windows = build_target_sequences(
            train_source,
            window_size=self.window_size,
            stride=self.stride,
            feature_columns=self.feature_columns_,
            target_mode=self.target_mode,
            prediction_horizon=self.prediction_horizon,
        )
        val_windows = build_target_sequences(
            val_source,
            window_size=self.window_size,
            stride=self.stride,
            feature_columns=self.feature_columns_,
            target_mode=self.target_mode,
            prediction_horizon=self.prediction_horizon,
        )

        self.scaler = fit_sequence_scaler(train_windows.features)
        train_features = transform_sequence_features(train_windows.features, self.scaler)
        val_features = transform_sequence_features(val_windows.features, self.scaler)

        self.train_dataset = SequenceTensorDataset(train_features, train_windows.targets)
        self.val_dataset = SequenceTensorDataset(val_features, val_windows.targets)
        self.train_metadata = train_windows.metadata
        self.val_metadata = val_windows.metadata
        self.target_name_ = train_windows.target_name

        if self.test_df is not None and stage in (None, "test", "predict", "fit"):
            test_source = self.test_df.copy()
            if "rul" not in test_source.columns:
                if self.test_rul_df is None:
                    raise ValueError("test_rul_df is required when test_df does not already contain row-level RUL.")
                test_source = add_test_rul(test_source, self.test_rul_df)

            test_windows = build_target_sequences(
                test_source,
                window_size=self.window_size,
                stride=self.stride,
                feature_columns=self.feature_columns_,
                target_mode=self.target_mode,
                prediction_horizon=self.prediction_horizon,
            )
            test_features = transform_sequence_features(test_windows.features, self.scaler)
            self.test_dataset = SequenceTensorDataset(test_features, test_windows.targets)
            self.test_metadata = test_windows.metadata

    def _dataloader(self, dataset: SequenceTensorDataset | None, shuffle: bool = False) -> DataLoader:
        if dataset is None:
            raise RuntimeError("The requested dataset has not been created. Call `setup()` first.")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False)

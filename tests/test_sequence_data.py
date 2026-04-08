from __future__ import annotations

import numpy as np
import pandas as pd

from src.sequence_data import (
    CMAPSSSequenceDataModule,
    add_test_rul,
    build_target_sequences,
    split_train_validation_by_unit,
)


def _make_train_df(num_units: int = 4, cycles_per_unit: int = 6) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for unit in range(1, num_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            row: dict[str, float | int] = {
                "unit": unit,
                "cycle": cycle,
                "setting_1": 0.1 * unit,
                "setting_2": 0.01 * cycle,
                "setting_3": 100.0,
            }
            for sensor_idx in range(1, 22):
                row[f"sensor_{sensor_idx}"] = float(unit * 100 + cycle + sensor_idx)
            rows.append(row)
    return pd.DataFrame(rows)


def test_add_test_rul_expands_unit_level_targets_to_rows() -> None:
    test_df = _make_train_df(num_units=2, cycles_per_unit=4)
    rul_df = pd.DataFrame({"rul": [3, 1]}, index=pd.Index([1, 2], name="unit"))

    with_rul = add_test_rul(test_df, rul_df)

    unit_1 = with_rul[with_rul["unit"] == 1]["rul"].tolist()
    unit_2 = with_rul[with_rul["unit"] == 2]["rul"].tolist()
    assert unit_1 == [6.0, 5.0, 4.0, 3.0]
    assert unit_2 == [4.0, 3.0, 2.0, 1.0]


def test_build_target_sequences_aligns_rul_to_window_end() -> None:
    train_df = _make_train_df(num_units=1, cycles_per_unit=5)
    train_df["rul"] = [4, 3, 2, 1, 0]

    sequence_set = build_target_sequences(train_df, window_size=3, stride=1, target_mode="rul")

    assert sequence_set.features.shape == (3, 3, 24)
    assert sequence_set.targets.tolist() == [2.0, 1.0, 0.0]
    assert sequence_set.metadata["target_cycle"].tolist() == [3, 4, 5]
    assert sequence_set.metadata["target_rul"].tolist() == [2.0, 1.0, 0.0]


def test_build_target_sequences_marks_failure_in_next_window() -> None:
    train_df = _make_train_df(num_units=1, cycles_per_unit=5)
    train_df["rul"] = [4, 3, 2, 1, 0]

    sequence_set = build_target_sequences(
        train_df,
        window_size=2,
        stride=1,
        target_mode="failure_in_next_window",
        prediction_horizon=2,
    )

    assert sequence_set.targets.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert sequence_set.metadata["failure_in_next_window"].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_split_train_validation_by_unit_keeps_units_disjoint() -> None:
    train_df = _make_train_df(num_units=5, cycles_per_unit=4)

    train_split, val_split = split_train_validation_by_unit(train_df, validation_fraction=0.4, random_state=7)

    assert set(train_split["unit"]).isdisjoint(set(val_split["unit"]))
    assert len(set(train_split["unit"]) | set(val_split["unit"])) == 5


def test_cmapss_sequence_datamodule_builds_scaled_batches() -> None:
    train_df = _make_train_df(num_units=4, cycles_per_unit=6)
    test_df = _make_train_df(num_units=2, cycles_per_unit=4)
    test_rul_df = pd.DataFrame({"rul": [2, 3]}, index=pd.Index([1, 2], name="unit"))

    datamodule = CMAPSSSequenceDataModule(
        train_df=train_df,
        test_df=test_df,
        test_rul_df=test_rul_df,
        target_mode="rul",
        window_size=3,
        stride=1,
        validation_fraction=0.25,
        batch_size=4,
    )
    datamodule.setup()

    train_batch_features, train_batch_targets = next(iter(datamodule.train_dataloader()))
    test_batch_features, test_batch_targets = next(iter(datamodule.test_dataloader()))

    assert train_batch_features.ndim == 3
    assert train_batch_features.shape[1:] == (3, 24)
    assert train_batch_targets.ndim == 1
    assert test_batch_features.ndim == 3
    assert test_batch_targets.ndim == 1

    train_array = datamodule.train_dataset.features.numpy()
    flattened = train_array.reshape(-1, train_array.shape[-1])
    assert np.allclose(flattened.mean(axis=0), 0.0, atol=1e-6)

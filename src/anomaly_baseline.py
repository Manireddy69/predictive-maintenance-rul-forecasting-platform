from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import OPERATIONAL_SETTING_COLUMNS, SENSOR_COLUMNS, load_dataset

ANOMALY_FEATURE_COLUMNS = [*OPERATIONAL_SETTING_COLUMNS, *SENSOR_COLUMNS]
ANOMALY_SCORE_COLUMNS = [
    "isolation_forest_score",
    "local_outlier_factor_score",
    "zscore_score",
    "mad_score",
]


def _load_pyod_lof():
    try:
        from pyod.models.lof import LOF
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "PyOD is not installed. Run `pip install -r requirements.txt` before training the LOF baseline."
        ) from exc
    return LOF


@dataclass
class AnomalyEvaluation:
    model_name: str
    roc_auc: float
    pr_auc: float
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray

    def pr_curve_frame(self) -> pd.DataFrame:
        threshold_values = np.concatenate(([np.nan], self.thresholds.astype(float, copy=False)))
        return pd.DataFrame(
            {
                "precision": self.precision.astype(float, copy=False),
                "recall": self.recall.astype(float, copy=False),
                "threshold": threshold_values,
            }
        )


@dataclass(frozen=True)
class StatisticalReference:
    center: np.ndarray
    scale: np.ndarray
    model_name: str


def prepare_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or ANOMALY_FEATURE_COLUMNS
    missing = set(feature_columns).difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Cannot build anomaly features without columns: {missing_str}")
    return df.loc[:, feature_columns].astype(float)


def _apply_injected_anomaly(
    row: pd.Series,
    rng: np.random.Generator,
    sensor_columns: list[str],
) -> tuple[pd.Series, str]:
    updated = row.copy()
    anomaly_type = str(rng.choice(["spike", "dropout", "drift_jump"]))

    if anomaly_type == "spike":
        affected = rng.choice(sensor_columns, size=min(4, len(sensor_columns)), replace=False)
        updated.loc[list(affected)] = updated.loc[list(affected)] + rng.normal(24.0, 4.5, size=len(affected))
    elif anomaly_type == "dropout":
        affected = rng.choice(sensor_columns, size=min(3, len(sensor_columns)), replace=False)
        updated.loc[list(affected)] = updated.loc[list(affected)] * rng.uniform(0.01, 0.08, size=len(affected))
    else:
        affected = rng.choice(sensor_columns, size=min(6, len(sensor_columns)), replace=False)
        updated.loc[list(affected)] = updated.loc[list(affected)] + rng.normal(10.0, 2.0, size=len(affected))

    return updated, anomaly_type


def build_holdout_with_injected_anomalies(
    holdout_df: pd.DataFrame,
    anomaly_fraction: float = 0.08,
    random_state: int | None = None,
    sensor_columns: list[str] | None = None,
    append_anomalies: bool = True,
) -> pd.DataFrame:
    if not 0 < anomaly_fraction < 1:
        raise ValueError("anomaly_fraction must be between 0 and 1.")
    if holdout_df.empty:
        raise ValueError("holdout_df must contain at least one row.")

    required_columns = {"unit", "cycle", *ANOMALY_FEATURE_COLUMNS}
    missing = required_columns.difference(holdout_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Cannot inject anomalies without columns: {missing_str}")

    rng = np.random.default_rng(random_state)
    sensor_columns = sensor_columns or SENSOR_COLUMNS

    clean_holdout = holdout_df.copy()
    clean_holdout["is_anomaly"] = 0
    clean_holdout["injected_anomaly_type"] = "normal"

    unit_max_cycle = clean_holdout.groupby("unit")["cycle"].transform("max")
    late_life_mask = clean_holdout["cycle"] >= np.ceil(unit_max_cycle * 0.7)
    candidate_index = clean_holdout.index[late_life_mask]
    if len(candidate_index) == 0:
        candidate_index = clean_holdout.index

    anomaly_count = max(1, int(round(len(clean_holdout) * anomaly_fraction)))
    anomaly_count = min(anomaly_count, len(candidate_index))
    selected_index = rng.choice(candidate_index.to_numpy(), size=anomaly_count, replace=False)

    if append_anomalies:
        anomalous_rows = clean_holdout.loc[selected_index].copy()
        anomaly_types: list[str] = []
        for index in anomalous_rows.index:
            updated_row, anomaly_type = _apply_injected_anomaly(
                anomalous_rows.loc[index, sensor_columns],
                rng=rng,
                sensor_columns=sensor_columns,
            )
            anomalous_rows.loc[index, sensor_columns] = updated_row
            anomaly_types.append(anomaly_type)

        anomalous_rows["is_anomaly"] = 1
        anomalous_rows["injected_anomaly_type"] = anomaly_types
        return pd.concat([clean_holdout, anomalous_rows], ignore_index=True)

    in_place_holdout = clean_holdout.copy()
    for index in np.sort(selected_index):
        updated_row, anomaly_type = _apply_injected_anomaly(
            in_place_holdout.loc[index, sensor_columns],
            rng=rng,
            sensor_columns=sensor_columns,
        )
        in_place_holdout.loc[index, sensor_columns] = updated_row
        in_place_holdout.loc[index, "is_anomaly"] = 1
        in_place_holdout.loc[index, "injected_anomaly_type"] = anomaly_type

    return in_place_holdout


def fit_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.08,
    random_state: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_train)
    return model


def _make_safe_scale(scale: np.ndarray) -> np.ndarray:
    safe_scale = np.asarray(scale, dtype=float).copy()
    safe_scale[~np.isfinite(safe_scale)] = 1.0
    safe_scale[np.abs(safe_scale) < 1e-12] = 1.0
    return safe_scale


def fit_zscore_reference(X_train: np.ndarray) -> StatisticalReference:
    center = np.mean(X_train, axis=0)
    scale = _make_safe_scale(np.std(X_train, axis=0, ddof=0))
    return StatisticalReference(center=center, scale=scale, model_name="Z-Score Distance")


def fit_mad_reference(X_train: np.ndarray) -> StatisticalReference:
    center = np.median(X_train, axis=0)
    mad = np.median(np.abs(X_train - center), axis=0)
    scale = _make_safe_scale(1.4826 * mad)
    return StatisticalReference(center=center, scale=scale, model_name="MAD Distance")


def score_statistical_reference(
    X_eval: np.ndarray,
    reference: StatisticalReference,
) -> np.ndarray:
    standardized = (X_eval - reference.center) / reference.scale
    return np.sqrt(np.mean(np.square(standardized), axis=1))


def fit_pyod_lof(
    X_train: np.ndarray,
    contamination: float = 0.08,
    n_neighbors: int = 35,
) -> object:
    LOF = _load_pyod_lof()
    model = LOF(contamination=contamination, n_neighbors=n_neighbors)
    model.fit(X_train)
    return model


def evaluate_anomaly_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str,
) -> AnomalyEvaluation:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    return AnomalyEvaluation(
        model_name=model_name,
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
        precision=precision,
        recall=recall,
        thresholds=thresholds,
    )


def score_anomaly_detectors(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    contamination: float = 0.08,
    random_state: int = 42,
    n_neighbors: int = 35,
) -> pd.DataFrame:
    X_train_frame = prepare_feature_frame(train_df)
    X_eval_frame = prepare_feature_frame(evaluation_df)

    X_train = X_train_frame.to_numpy(dtype=float, copy=False)
    X_eval = X_eval_frame.to_numpy(dtype=float, copy=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    isolation_forest = fit_isolation_forest(
        X_train_scaled,
        contamination=contamination,
        random_state=random_state,
    )
    if_scores = -isolation_forest.decision_function(X_eval_scaled)

    lof_model = fit_pyod_lof(
        X_train_scaled,
        contamination=contamination,
        n_neighbors=n_neighbors,
    )
    lof_scores = np.asarray(lof_model.decision_function(X_eval_scaled), dtype=float)

    zscore_reference = fit_zscore_reference(X_train)
    zscore_scores = score_statistical_reference(X_eval, zscore_reference)

    mad_reference = fit_mad_reference(X_train)
    mad_scores = score_statistical_reference(X_eval, mad_reference)

    scored_evaluation_df = evaluation_df.copy()
    scored_evaluation_df["isolation_forest_score"] = if_scores.astype(float, copy=False)
    scored_evaluation_df["local_outlier_factor_score"] = lof_scores.astype(float, copy=False)
    scored_evaluation_df["zscore_score"] = zscore_scores.astype(float, copy=False)
    scored_evaluation_df["mad_score"] = mad_scores.astype(float, copy=False)
    return scored_evaluation_df


def evaluate_scored_anomaly_detectors(scored_evaluation_df: pd.DataFrame) -> dict[str, AnomalyEvaluation]:
    if "is_anomaly" not in scored_evaluation_df.columns:
        raise ValueError("scored_evaluation_df must include an is_anomaly label column for evaluation.")

    y_true = scored_evaluation_df["is_anomaly"].to_numpy(dtype=int)
    return {
        "isolation_forest": evaluate_anomaly_scores(
            y_true,
            scored_evaluation_df["isolation_forest_score"].to_numpy(dtype=float, copy=False),
            model_name="Isolation Forest",
        ),
        "local_outlier_factor": evaluate_anomaly_scores(
            y_true,
            scored_evaluation_df["local_outlier_factor_score"].to_numpy(dtype=float, copy=False),
            model_name="Local Outlier Factor",
        ),
        "zscore": evaluate_anomaly_scores(
            y_true,
            scored_evaluation_df["zscore_score"].to_numpy(dtype=float, copy=False),
            model_name="Z-Score Distance",
        ),
        "mad": evaluate_anomaly_scores(
            y_true,
            scored_evaluation_df["mad_score"].to_numpy(dtype=float, copy=False),
            model_name="MAD Distance",
        ),
    }


def evaluation_summary_frame(results: dict[str, AnomalyEvaluation]) -> pd.DataFrame:
    rows = [
        {
            "model_key": model_key,
            "model_name": result.model_name,
            "roc_auc": result.roc_auc,
            "pr_auc": result.pr_auc,
            "pr_curve_points": len(result.precision),
        }
        for model_key, result in results.items()
    ]
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def run_scored_baseline_experiment(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    anomaly_fraction: float = 0.08,
    contamination: float = 0.08,
    random_state: int = 42,
    n_neighbors: int = 35,
    append_anomalies: bool = True,
) -> tuple[pd.DataFrame, dict[str, AnomalyEvaluation]]:
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")

    evaluation_df = build_holdout_with_injected_anomalies(
        holdout_df,
        anomaly_fraction=anomaly_fraction,
        random_state=random_state,
        append_anomalies=append_anomalies,
    )
    scored_evaluation_df = score_anomaly_detectors(
        train_df=train_df,
        evaluation_df=evaluation_df,
        contamination=contamination,
        random_state=random_state,
        n_neighbors=n_neighbors,
    )
    results = evaluate_scored_anomaly_detectors(scored_evaluation_df)
    return scored_evaluation_df, results


def run_baseline_experiment(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    anomaly_fraction: float = 0.08,
    contamination: float = 0.08,
    random_state: int = 42,
    n_neighbors: int = 35,
) -> tuple[pd.DataFrame, dict[str, AnomalyEvaluation]]:
    scored_evaluation_df, results = run_scored_baseline_experiment(
        train_df=train_df,
        holdout_df=holdout_df,
        anomaly_fraction=anomaly_fraction,
        contamination=contamination,
        random_state=random_state,
        n_neighbors=n_neighbors,
    )
    evaluation_df = scored_evaluation_df.drop(columns=ANOMALY_SCORE_COLUMNS)
    return evaluation_df, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline anomaly detectors on synthetic normal telemetry.")
    parser.add_argument("--synthetic-units", type=int, default=40, help="Total synthetic units across train and hold-out.")
    parser.add_argument(
        "--synthetic-test-fraction",
        type=float,
        default=0.25,
        help="Fraction of units reserved for the hold-out split.",
    )
    parser.add_argument(
        "--holdout-anomaly-fraction",
        type=float,
        default=0.08,
        help="Fraction of hold-out rows duplicated and perturbed into labeled anomalies.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.08,
        help="Expected anomaly rate used by both baseline detectors.",
    )
    parser.add_argument("--n-neighbors", type=int, default=35, help="Neighborhood size for the PyOD LOF baseline.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible results.")
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

    evaluation_df, results = run_baseline_experiment(
        train_df=train_df,
        holdout_df=holdout_df,
        anomaly_fraction=args.holdout_anomaly_fraction,
        contamination=args.contamination,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
    )

    print("=== Baseline anomaly detection experiment ===")
    print(f"Normal-only train rows: {len(train_df)}")
    print(f"Hold-out rows before injection: {len(holdout_df)}")
    print(f"Hold-out rows after injection:  {len(evaluation_df)}")
    print(f"Labeled anomalies in hold-out: {int(evaluation_df['is_anomaly'].sum())}")
    print()

    for result in results.values():
        print(f"{result.model_name}:")
        print(f"  ROC-AUC: {result.roc_auc:.4f}")
        print(f"  PR-AUC:  {result.pr_auc:.4f}")
        print(f"  PR curve points: {len(result.precision)}")
        print()


if __name__ == "__main__":
    main()

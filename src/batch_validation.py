from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data import SENSOR_COLUMNS

DEFAULT_SENSOR_MIN = 0.0
DEFAULT_SENSOR_MAX = 2000.0


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    success: bool
    details: str


@dataclass(frozen=True)
class ValidationReport:
    success: bool
    engine: str
    checks: tuple[ValidationCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "engine": str(self.engine),
            "checks": [
                {
                    "name": str(check.name),
                    "success": bool(check.success),
                    "details": str(check.details),
                }
                for check in self.checks
            ],
        }

    def to_error_message(self) -> str:
        failed_checks = [check for check in self.checks if not check.success]
        summary = "; ".join(f"{check.name}: {check.details}" for check in failed_checks)
        return f"Batch validation failed. {summary}"


def validate_sensor_batch(
    df: pd.DataFrame,
    sensor_columns: list[str] | tuple[str, ...] | None = None,
    sensor_min: float = DEFAULT_SENSOR_MIN,
    sensor_max: float = DEFAULT_SENSOR_MAX,
) -> ValidationReport:
    sensor_columns = list(sensor_columns or SENSOR_COLUMNS)
    gx_report = _try_great_expectations_validation(
        df=df,
        sensor_columns=sensor_columns,
        sensor_min=sensor_min,
        sensor_max=sensor_max,
    )
    if gx_report is not None:
        return gx_report

    checks = _native_validation_checks(
        df=df,
        sensor_columns=sensor_columns,
        sensor_min=sensor_min,
        sensor_max=sensor_max,
    )
    return ValidationReport(
        success=all(check.success for check in checks),
        engine="native-fallback",
        checks=tuple(checks),
    )


def _native_validation_checks(
    df: pd.DataFrame,
    sensor_columns: list[str],
    sensor_min: float,
    sensor_max: float,
) -> list[ValidationCheck]:
    required_columns = ["event_time", "equipment_id", *sensor_columns]
    missing_columns = sorted(set(required_columns).difference(df.columns))
    parsed_event_time = pd.to_datetime(df.get("event_time"), utc=True, errors="coerce")
    key_columns = ["event_time", "equipment_id", "run_id"] if "run_id" in df.columns else ["event_time", "equipment_id"]

    checks = [
        ValidationCheck(
            name="required_columns",
            success=not missing_columns,
            details="all required columns present" if not missing_columns else f"missing columns: {', '.join(missing_columns)}",
        ),
        ValidationCheck(
            name="event_time_parseable",
            success=parsed_event_time.notna().all(),
            details="event_time values are parseable" if parsed_event_time.notna().all() else "event_time contains unparseable values",
        ),
        ValidationCheck(
            name="key_columns_non_null",
            success=not df[key_columns].isna().any().any() if set(key_columns).issubset(df.columns) else False,
            details="key columns are non-null" if set(key_columns).issubset(df.columns) and not df[key_columns].isna().any().any() else "one or more key columns are null",
        ),
        ValidationCheck(
            name="unique_batch_keys",
            success=not df.duplicated(subset=key_columns).any() if set(key_columns).issubset(df.columns) else False,
            details="batch keys are unique" if set(key_columns).issubset(df.columns) and not df.duplicated(subset=key_columns).any() else "duplicate event_time/equipment keys detected",
        ),
    ]

    for sensor_name in sensor_columns:
        if sensor_name not in df.columns:
            continue
        sensor_values = pd.to_numeric(df[sensor_name], errors="coerce")
        checks.append(
            ValidationCheck(
                name=f"{sensor_name}_numeric",
                success=sensor_values.notna().all(),
                details="numeric values only" if sensor_values.notna().all() else "contains non-numeric values",
            )
        )
        checks.append(
            ValidationCheck(
                name=f"{sensor_name}_range",
                success=sensor_values.between(sensor_min, sensor_max).all(),
                details=f"values within [{sensor_min}, {sensor_max}]" if sensor_values.between(sensor_min, sensor_max).all() else f"values outside [{sensor_min}, {sensor_max}]",
            )
        )

    return checks


def _try_great_expectations_validation(
    df: pd.DataFrame,
    sensor_columns: list[str],
    sensor_min: float,
    sensor_max: float,
) -> ValidationReport | None:
    try:
        from great_expectations.dataset import PandasDataset
    except ImportError:
        return None
    except Exception:
        return None

    dataset = PandasDataset(df.copy())
    dataset["event_time"] = pd.to_datetime(dataset["event_time"], utc=True, errors="coerce")

    results: list[ValidationCheck] = []
    required_columns = ["event_time", "equipment_id", *sensor_columns]
    key_columns = ["event_time", "equipment_id", "run_id"] if "run_id" in dataset.columns else ["event_time", "equipment_id"]
    missing_columns = sorted(set(required_columns).difference(dataset.columns))

    checks = [
        ("event_time_non_null", dataset.expect_column_values_to_not_be_null("event_time")),
        ("equipment_id_non_null", dataset.expect_column_values_to_not_be_null("equipment_id")),
        ("unique_batch_keys", dataset.expect_compound_columns_to_be_unique(key_columns)),
    ]

    results.append(
        ValidationCheck(
            name="required_columns",
            success=not missing_columns,
            details="all required columns present" if not missing_columns else f"missing columns: {', '.join(missing_columns)}",
        )
    )

    for name, result in checks:
        results.append(
            ValidationCheck(
                name=name,
                success=bool(result["success"]),
                details=str(result.get("result", {})),
            )
        )

    for sensor_name in sensor_columns:
        if sensor_name not in dataset.columns:
            continue
        numeric_result = dataset.expect_column_values_to_not_be_null(sensor_name)
        range_result = dataset.expect_column_values_to_be_between(
            sensor_name,
            min_value=sensor_min,
            max_value=sensor_max,
        )
        results.append(
            ValidationCheck(
                name=f"{sensor_name}_non_null",
                success=bool(numeric_result["success"]),
                details=str(numeric_result.get("result", {})),
            )
        )
        results.append(
            ValidationCheck(
                name=f"{sensor_name}_range",
                success=bool(range_result["success"]),
                details=str(range_result.get("result", {})),
            )
        )

    return ValidationReport(
        success=all(check.success for check in results),
        engine="great-expectations",
        checks=tuple(results),
    )

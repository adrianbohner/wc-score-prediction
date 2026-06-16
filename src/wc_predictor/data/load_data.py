from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_predictor.data.validate_data import (
    GOALSCORERS_REQUIRED_COLUMNS,
    RESULTS_REQUIRED_COLUMNS,
    DataValidationError,
    add_match_id,
    require_columns,
    validate_goalscorers,
    validate_results,
)


def load_results(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path, "results.csv")
    require_columns(df, RESULTS_REQUIRED_COLUMNS, "results.csv")
    df = _parse_date_column(df, "date", "results.csv")
    df = _coerce_score_columns(df)
    df = _coerce_boolean_columns(df, ["neutral"], "results.csv")
    validate_results(df)
    return add_match_id(df)


def load_goalscorers(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path, "goalscorers.csv")
    require_columns(df, GOALSCORERS_REQUIRED_COLUMNS, "goalscorers.csv")
    df = _parse_date_column(df, "date", "goalscorers.csv")
    df = _coerce_boolean_columns(df, ["own_goal", "penalty"], "goalscorers.csv")
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    validate_goalscorers(df)
    return df


def _read_csv(path: str | Path, dataset_name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(Path(path))
    except FileNotFoundError as exc:
        raise DataValidationError(f"{dataset_name} was not found at {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"{dataset_name} is empty") from exc


def _parse_date_column(df: pd.DataFrame, column: str, dataset_name: str) -> pd.DataFrame:
    if column not in df.columns:
        return df

    result = df.copy()
    parsed = pd.to_datetime(result[column], errors="coerce")
    if parsed.isna().any():
        raise DataValidationError(f"{dataset_name}.{column} contains invalid dates")
    result[column] = parsed
    return result


def _coerce_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ["home_score", "away_score"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _coerce_boolean_columns(
    df: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            continue
        result[column] = result[column].map(lambda value: _coerce_bool(value, column, dataset_name))
    return result


def _coerce_bool(value: object, column: str, dataset_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        raise DataValidationError(f"{dataset_name}.{column} contains missing boolean values")

    normalized = str(value).lower().strip()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False

    raise DataValidationError(f"{dataset_name}.{column} contains invalid boolean value: {value}")

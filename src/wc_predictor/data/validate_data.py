from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


RESULTS_REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}

GOALSCORERS_REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "team",
    "scorer",
    "minute",
    "own_goal",
    "penalty",
}


class DataValidationError(ValueError):
    """Raised when an input dataset cannot be used safely."""


def require_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(f"{dataset_name} is missing required columns: {missing_text}")


def validate_results(df: pd.DataFrame) -> None:
    require_columns(df, RESULTS_REQUIRED_COLUMNS, "results.csv")
    _validate_datetime_column(df, "date", "results.csv")
    _validate_team_columns(df, ["home_team", "away_team"], "results.csv")
    _validate_score_pair(df)
    _validate_boolean_column(df, "neutral", "results.csv")


def validate_goalscorers(df: pd.DataFrame) -> None:
    require_columns(df, GOALSCORERS_REQUIRED_COLUMNS, "goalscorers.csv")
    _validate_datetime_column(df, "date", "goalscorers.csv")
    _validate_team_columns(df, ["home_team", "away_team", "team"], "goalscorers.csv")
    _validate_boolean_column(df, "own_goal", "goalscorers.csv")
    _validate_boolean_column(df, "penalty", "goalscorers.csv")

    if df["minute"].notna().any():
        invalid_minutes = pd.to_numeric(df["minute"], errors="coerce").isna() & df["minute"].notna()
        if invalid_minutes.any():
            raise DataValidationError("goalscorers.csv contains non-numeric minute values")


def add_match_id(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    dates = result["date"].dt.strftime("%Y-%m-%d")
    city = result["city"].fillna("").astype(str).str.strip()
    result["match_id"] = (
        dates
        + "|"
        + result["home_team"].astype(str).str.strip()
        + "|"
        + result["away_team"].astype(str).str.strip()
        + "|"
        + city
    )
    return result


def find_duplicate_match_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "match_id" not in df.columns:
        df = add_match_id(df)
    duplicate_mask = df["match_id"].duplicated(keep=False)
    return df.loc[duplicate_mask].sort_values("match_id").copy()


def get_completed_matches(df: pd.DataFrame) -> pd.DataFrame:
    completed_mask = df["home_score"].notna() & df["away_score"].notna()
    return df.loc[completed_mask].copy()


def _validate_datetime_column(df: pd.DataFrame, column: str, dataset_name: str) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        raise DataValidationError(f"{dataset_name}.{column} must be parsed as datetime")
    if df[column].isna().any():
        raise DataValidationError(f"{dataset_name}.{column} contains invalid or missing dates")


def _validate_team_columns(df: pd.DataFrame, columns: list[str], dataset_name: str) -> None:
    for column in columns:
        values = df[column]
        invalid = values.isna() | values.astype(str).str.strip().eq("")
        if invalid.any():
            raise DataValidationError(f"{dataset_name}.{column} contains empty team names")


def _validate_score_pair(df: pd.DataFrame) -> None:
    home_missing = df["home_score"].isna()
    away_missing = df["away_score"].isna()
    incomplete_rows = home_missing ^ away_missing
    if incomplete_rows.any():
        raise DataValidationError(
            "results.csv rows must have both scores present or both scores missing"
        )

    completed = get_completed_matches(df)
    if completed.empty:
        return

    scores = completed[["home_score", "away_score"]]
    if (scores < 0).any().any():
        raise DataValidationError("results.csv contains negative scores")

    score_values = scores.to_numpy(dtype=float)
    if not np.isclose(score_values, np.round(score_values)).all():
        raise DataValidationError("results.csv completed scores must be whole numbers")


def _validate_boolean_column(df: pd.DataFrame, column: str, dataset_name: str) -> None:
    invalid = df[column].isna()
    if invalid.any():
        raise DataValidationError(f"{dataset_name}.{column} contains missing boolean values")

    bool_types = pd.api.types.is_bool_dtype(df[column])
    if bool_types:
        return

    allowed = {"true", "false", "1", "0", "yes", "no"}
    as_text = df[column].astype(str).str.lower().str.strip()
    if ~as_text.isin(allowed).all():
        raise DataValidationError(f"{dataset_name}.{column} contains invalid boolean values")


from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wc_predictor.data import DataValidationError


DEFAULT_FEATURE_COLUMNS = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff",
    "elo_abs_diff",
    "home_points_avg_5",
    "away_points_avg_5",
    "home_points_avg_10",
    "away_points_avg_10",
    "home_goals_for_avg_5",
    "away_goals_for_avg_5",
    "home_goals_for_avg_10",
    "away_goals_for_avg_10",
    "home_goals_against_avg_5",
    "away_goals_against_avg_5",
    "home_goals_against_avg_10",
    "away_goals_against_avg_10",
    "home_goal_diff_avg_10",
    "away_goal_diff_avg_10",
    "home_attack_strength_10",
    "away_attack_strength_10",
    "home_defense_strength_10",
    "away_defense_strength_10",
    "home_win_rate_5",
    "away_win_rate_5",
    "home_win_rate_10",
    "away_win_rate_10",
    "home_unbeaten_rate_10",
    "away_unbeaten_rate_10",
    "home_clean_sheet_rate_10",
    "away_clean_sheet_rate_10",
    "home_scored_rate_10",
    "away_scored_rate_10",
    "home_days_since_match",
    "away_days_since_match",
    "home_matches_last_30",
    "away_matches_last_30",
    "home_matches_last_365",
    "away_matches_last_365",
    "home_matches_available_10",
    "away_matches_available_10",
    "home_low_history_flag",
    "away_low_history_flag",
    "points_avg_5_diff",
    "points_avg_10_diff",
    "goals_for_avg_10_diff",
    "goals_against_avg_10_diff",
    "goal_diff_avg_10_diff",
    "attack_strength_10_diff",
    "defense_strength_10_diff",
    "win_rate_10_diff",
    "unbeaten_rate_10_diff",
    "clean_sheet_rate_10_diff",
    "scored_rate_10_diff",
    "days_since_match_diff",
    "matches_last_30_diff",
    "matches_last_365_diff",
    "is_neutral",
    "home_host_advantage",
    "away_host_advantage",
    "is_world_cup",
    "is_friendly",
    "is_qualifier",
]


@dataclass
class SimplePoissonScoreModel:
    max_goals: int = 8
    alpha: float = 1.0
    max_iter: int = 1000
    min_expected_goals: float = 0.05
    max_expected_goals: float = 6.0
    feature_columns: list[str] | None = None

    def fit(
        self,
        training_features: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> "SimplePoissonScoreModel":
        self.feature_columns_ = list(self.feature_columns or DEFAULT_FEATURE_COLUMNS)
        _require_columns(training_features, self.feature_columns_)
        _require_columns(training_features, ["home_score", "away_score"])

        X = self._prepare_features(training_features)
        y_home = _prepare_target(training_features["home_score"], "home_score")
        y_away = _prepare_target(training_features["away_score"], "away_score")

        fit_params = {} if sample_weight is None else {"model__sample_weight": sample_weight}
        self.home_model_ = self._build_regressor()
        self.away_model_ = self._build_regressor()
        self.home_model_.fit(X, y_home, **fit_params)
        self.away_model_.fit(X, y_away, **fit_params)
        return self

    def predict_expected_goals(self, features: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        X = self._prepare_features(features)
        home_goals = self.home_model_.predict(X)
        away_goals = self.away_model_.predict(X)

        return pd.DataFrame(
            {
                "expected_home_goals": np.clip(
                    home_goals, self.min_expected_goals, self.max_expected_goals
                ),
                "expected_away_goals": np.clip(
                    away_goals, self.min_expected_goals, self.max_expected_goals
                ),
            },
            index=features.index,
        )

    def predict_score_matrix(self, features: pd.DataFrame | pd.Series) -> np.ndarray:
        feature_frame = _as_frame(features)
        expected = self.predict_expected_goals(feature_frame).iloc[0]
        return score_matrix_from_expected_goals(
            expected_home_goals=float(expected["expected_home_goals"]),
            expected_away_goals=float(expected["expected_away_goals"]),
            max_goals=self.max_goals,
        )

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        rows: list[dict[str, Any]] = []
        expected_goals = self.predict_expected_goals(features)

        for index, expected in expected_goals.iterrows():
            matrix = score_matrix_from_expected_goals(
                expected_home_goals=float(expected["expected_home_goals"]),
                expected_away_goals=float(expected["expected_away_goals"]),
                max_goals=self.max_goals,
            )
            outcome_probs = outcome_probabilities(matrix)
            scoreline_options = top_scorelines(matrix, count=5)
            most_likely = scoreline_options[0]

            rows.append(
                {
                    "expected_home_goals": float(expected["expected_home_goals"]),
                    "expected_away_goals": float(expected["expected_away_goals"]),
                    "pred_home_goals": most_likely["home_goals"],
                    "pred_away_goals": most_likely["away_goals"],
                    "most_likely_score_prob": most_likely["probability"],
                    "top_scorelines": scoreline_options,
                    **outcome_probs,
                }
            )

        return pd.DataFrame(rows, index=features.index)

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        _require_columns(features, self.feature_columns_)
        selected = features[self.feature_columns_].copy()
        numeric = selected.apply(pd.to_numeric, errors="coerce")
        invalid = numeric.isna()
        if invalid.any().any():
            bad_columns = sorted(invalid.columns[invalid.any()].tolist())
            raise DataValidationError(
                "model features contain missing or non-numeric values: "
                + ", ".join(bad_columns)
            )
        return numeric.astype(float)

    def _build_regressor(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    PoissonRegressor(
                        alpha=self.alpha,
                        max_iter=self.max_iter,
                    ),
                ),
            ]
        )

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "home_model_") or not hasattr(self, "away_model_"):
            raise DataValidationError("SimplePoissonScoreModel must be fitted before prediction")


def score_matrix_from_expected_goals(
    expected_home_goals: float,
    expected_away_goals: float,
    max_goals: int = 8,
) -> np.ndarray:
    if expected_home_goals <= 0 or expected_away_goals <= 0:
        raise DataValidationError("expected goals must be positive")

    home_probs = np.array(
        [_poisson_pmf(goal_count, expected_home_goals) for goal_count in range(max_goals + 1)]
    )
    away_probs = np.array(
        [_poisson_pmf(goal_count, expected_away_goals) for goal_count in range(max_goals + 1)]
    )
    matrix = np.outer(home_probs, away_probs)
    total = matrix.sum()
    if total <= 0:
        raise DataValidationError("score matrix has zero probability mass")
    return matrix / total


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    return {
        "prob_home_win": float(np.tril(matrix, k=-1).sum()),
        "prob_draw": float(np.trace(matrix)),
        "prob_away_win": float(np.triu(matrix, k=1).sum()),
    }


def top_scorelines(matrix: np.ndarray, count: int = 5) -> list[dict[str, float | int | str]]:
    flat_indices = np.argsort(matrix.ravel())[::-1][:count]
    scorelines = []
    for flat_index in flat_indices:
        home_goals, away_goals = np.unravel_index(flat_index, matrix.shape)
        scorelines.append(
            {
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
                "score": f"{home_goals}-{away_goals}",
                "probability": float(matrix[home_goals, away_goals]),
            }
        )
    return scorelines


def _poisson_pmf(goal_count: int, expected_goals: float) -> float:
    return exp(-expected_goals) * (expected_goals**goal_count) / factorial(goal_count)


def _prepare_target(target: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.isna().any():
        raise DataValidationError(f"{column_name} contains missing target values")
    if (numeric < 0).any():
        raise DataValidationError(f"{column_name} contains negative target values")
    return numeric


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise DataValidationError(f"missing model columns: {', '.join(missing)}")


def _as_frame(features: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(features, pd.Series):
        return features.to_frame().T
    return features

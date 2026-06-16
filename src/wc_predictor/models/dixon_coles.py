from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from wc_predictor.data import DataValidationError
from wc_predictor.models.poisson_baseline import (
    DEFAULT_FEATURE_COLUMNS,
    SimplePoissonScoreModel,
    outcome_probabilities,
    score_matrix_from_expected_goals,
    top_scorelines,
)

EPSILON = 1e-15


@dataclass
class DixonColesScoreModel:
    max_goals: int = 8
    alpha: float = 1.0
    rho_bounds: tuple[float, float] = (-0.3, 0.3)
    feature_columns: list[str] | None = None

    def fit(
        self,
        training_features: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> "DixonColesScoreModel":
        self.base_model_ = SimplePoissonScoreModel(
            max_goals=self.max_goals,
            alpha=self.alpha,
            feature_columns=self.feature_columns or DEFAULT_FEATURE_COLUMNS,
        ).fit(training_features, sample_weight=sample_weight)
        expected = self.base_model_.predict_expected_goals(training_features)
        self.rho_ = optimize_rho(
            expected_home_goals=expected["expected_home_goals"].to_numpy(dtype=float),
            expected_away_goals=expected["expected_away_goals"].to_numpy(dtype=float),
            actual_home_goals=training_features["home_score"].to_numpy(dtype=float),
            actual_away_goals=training_features["away_score"].to_numpy(dtype=float),
            max_goals=self.max_goals,
            rho_bounds=self.rho_bounds,
            sample_weight=sample_weight,
        )
        return self

    def predict_expected_goals(self, features: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        return self.base_model_.predict_expected_goals(features)

    def predict_score_matrix(self, features: pd.DataFrame | pd.Series) -> np.ndarray:
        self._check_is_fitted()
        feature_frame = features.to_frame().T if isinstance(features, pd.Series) else features
        expected = self.predict_expected_goals(feature_frame).iloc[0]
        return dixon_coles_score_matrix_from_expected_goals(
            expected_home_goals=float(expected["expected_home_goals"]),
            expected_away_goals=float(expected["expected_away_goals"]),
            rho=self.rho_,
            max_goals=self.max_goals,
        )

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        rows: list[dict[str, Any]] = []
        expected_goals = self.predict_expected_goals(features)

        for _, expected in expected_goals.iterrows():
            matrix = dixon_coles_score_matrix_from_expected_goals(
                expected_home_goals=float(expected["expected_home_goals"]),
                expected_away_goals=float(expected["expected_away_goals"]),
                rho=self.rho_,
                max_goals=self.max_goals,
            )
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
                    **outcome_probabilities(matrix),
                }
            )

        return pd.DataFrame(rows, index=features.index)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "base_model_") or not hasattr(self, "rho_"):
            raise DataValidationError("DixonColesScoreModel must be fitted before prediction")


def dixon_coles_score_matrix_from_expected_goals(
    expected_home_goals: float,
    expected_away_goals: float,
    rho: float,
    max_goals: int = 8,
) -> np.ndarray:
    base_matrix = score_matrix_from_expected_goals(
        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,
        max_goals=max_goals,
    )
    adjusted = base_matrix.copy()

    for home_goals in [0, 1]:
        for away_goals in [0, 1]:
            adjusted[home_goals, away_goals] *= dixon_coles_tau(
                home_goals=home_goals,
                away_goals=away_goals,
                expected_home_goals=expected_home_goals,
                expected_away_goals=expected_away_goals,
                rho=rho,
            )

    adjusted = np.clip(adjusted, 0.0, None)
    total = adjusted.sum()
    if total <= 0:
        raise DataValidationError("Dixon-Coles score matrix has zero probability mass")
    return adjusted / total


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    expected_home_goals: float,
    expected_away_goals: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - expected_home_goals * expected_away_goals * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + expected_home_goals * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + expected_away_goals * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def optimize_rho(
    expected_home_goals: np.ndarray,
    expected_away_goals: np.ndarray,
    actual_home_goals: np.ndarray,
    actual_away_goals: np.ndarray,
    max_goals: int,
    rho_bounds: tuple[float, float],
    sample_weight: np.ndarray | None = None,
) -> float:
    weights = np.ones(len(expected_home_goals)) if sample_weight is None else sample_weight

    def objective(rho: float) -> float:
        losses = []
        for home_lambda, away_lambda, home_goals, away_goals in zip(
            expected_home_goals,
            expected_away_goals,
            actual_home_goals,
            actual_away_goals,
        ):
            matrix = dixon_coles_score_matrix_from_expected_goals(
                expected_home_goals=float(home_lambda),
                expected_away_goals=float(away_lambda),
                rho=float(rho),
                max_goals=max_goals,
            )
            probability = _score_probability(matrix, home_goals, away_goals)
            losses.append(-np.log(max(probability, EPSILON)))
        return float(np.average(losses, weights=weights))

    result = minimize_scalar(objective, bounds=rho_bounds, method="bounded")
    if not result.success or not np.isfinite(result.x):
        return 0.0
    return float(np.clip(result.x, rho_bounds[0], rho_bounds[1]))


def _score_probability(matrix: np.ndarray, home_goals: float, away_goals: float) -> float:
    home = int(home_goals)
    away = int(away_goals)
    if home < 0 or away < 0:
        return EPSILON
    if home >= matrix.shape[0] or away >= matrix.shape[1]:
        return EPSILON
    return float(max(matrix[home, away], EPSILON))

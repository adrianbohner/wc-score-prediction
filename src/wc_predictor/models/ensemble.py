from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from wc_predictor.data import DataValidationError
from wc_predictor.models.poisson_baseline import outcome_probabilities, top_scorelines


@dataclass
class EnsembleScoreModel:
    """Weighted average of score-matrix models.

    Combines predictions by averaging per-match score matrices before computing
    outcome probabilities. This preserves each sub-model's low-score correction
    (e.g. Dixon-Coles rho) in the blended distribution.
    """

    models: list[Any]
    weights: list[float] | None = None
    max_goals: int = 8
    min_expected_goals: float = 0.05
    max_expected_goals: float = 6.0

    def __post_init__(self) -> None:
        if not self.models:
            raise DataValidationError("EnsembleScoreModel requires at least one sub-model")
        n = len(self.models)
        if self.weights is None:
            self.weights_ = [1.0 / n] * n
        else:
            if len(self.weights) != n:
                raise DataValidationError(
                    f"weights length {len(self.weights)} != number of models {n}"
                )
            total = sum(self.weights)
            if total <= 0:
                raise DataValidationError("ensemble weights must sum to a positive value")
            self.weights_ = [w / total for w in self.weights]

    def predict_expected_goals(self, features: pd.DataFrame) -> pd.DataFrame:
        home_goals = np.zeros(len(features))
        away_goals = np.zeros(len(features))
        for model, w in zip(self.models, self.weights_):
            eg = model.predict_expected_goals(features)
            home_goals += w * eg["expected_home_goals"].to_numpy(dtype=float)
            away_goals += w * eg["expected_away_goals"].to_numpy(dtype=float)
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
        if isinstance(features, pd.Series):
            features = features.to_frame().T.reset_index(drop=True)
        return self._averaged_matrix(features.iloc[[0]].reset_index(drop=True))

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        exp_goals = self.predict_expected_goals(features)
        rows: list[dict[str, Any]] = []
        for i in range(len(features)):
            row_frame = features.iloc[[i]].reset_index(drop=True)
            matrix = self._averaged_matrix(row_frame)
            scoreline_options = top_scorelines(matrix, count=5)
            most_likely = scoreline_options[0]
            rows.append(
                {
                    "expected_home_goals": float(exp_goals.iloc[i]["expected_home_goals"]),
                    "expected_away_goals": float(exp_goals.iloc[i]["expected_away_goals"]),
                    "pred_home_goals": most_likely["home_goals"],
                    "pred_away_goals": most_likely["away_goals"],
                    "most_likely_score_prob": most_likely["probability"],
                    "top_scorelines": scoreline_options,
                    **outcome_probabilities(matrix),
                }
            )
        return pd.DataFrame(rows, index=features.index)

    def _averaged_matrix(self, row_frame: pd.DataFrame) -> np.ndarray:
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        for model, w in zip(self.models, self.weights_):
            m = model.predict_score_matrix(row_frame)
            sz = min(self.max_goals + 1, m.shape[0])
            matrix[:sz, :sz] += w * m[:sz, :sz]
        total = matrix.sum()
        return matrix / total if total > 0 else matrix

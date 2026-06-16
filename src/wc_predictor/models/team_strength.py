from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc_predictor.data import DataValidationError
from wc_predictor.models.dixon_coles import (
    dixon_coles_score_matrix_from_expected_goals,
    optimize_rho,
)
from wc_predictor.models.poisson_baseline import (
    outcome_probabilities,
    score_matrix_from_expected_goals,
    top_scorelines,
)

_REQUIRED_FIT_COLS = [
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_host_advantage",
    "away_host_advantage",
]
_REQUIRED_PRED_COLS = ["home_team", "away_team", "home_host_advantage", "away_host_advantage"]


@dataclass
class TeamStrengthPoissonModel:
    """MLE team-strength Poisson model with optional Dixon-Coles low-score correction.

    Per-match parametrisation (home team i, away team j):
        log(lambda_home) = mu + h * venue + attack_i - defense_j
        log(lambda_away) = mu - h * venue + attack_j - defense_i

    where venue = +1 if home team has a real venue advantage, -1 if away team does, 0 for
    neutral matches. L2 regularisation on attack and defense naturally shrinks sparse teams toward the
    global mean, removing the need for the crude low_history_flag fallback.
    """

    max_goals: int = 8
    reg_strength: float = 0.01
    use_dc_correction: bool = True
    rho_bounds: tuple[float, float] = (-0.3, 0.3)
    max_iter: int = 2000
    min_expected_goals: float = 0.05
    max_expected_goals: float = 6.0

    def fit(
        self,
        training_features: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> "TeamStrengthPoissonModel":
        _require_cols(training_features, _REQUIRED_FIT_COLS)

        teams = sorted(
            set(training_features["home_team"].tolist() + training_features["away_team"].tolist())
        )
        self.teams_ = teams
        self.team_index_ = {t: i for i, t in enumerate(teams)}
        N = len(teams)

        home_idx = training_features["home_team"].map(self.team_index_).to_numpy(dtype=int)
        away_idx = training_features["away_team"].map(self.team_index_).to_numpy(dtype=int)
        home_scores = training_features["home_score"].to_numpy(dtype=float)
        away_scores = training_features["away_score"].to_numpy(dtype=float)
        venue_adv = _venue_multiplier(training_features)
        weights = (
            np.ones(len(training_features), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )

        mu_0 = np.log(max(float(np.mean(np.concatenate([home_scores, away_scores]))), 0.1))
        x0 = np.zeros(2 + 2 * N)
        x0[0] = mu_0
        x0[1] = 0.1

        result = minimize(
            _nll_and_grad,
            x0,
            args=(home_idx, away_idx, home_scores, away_scores, venue_adv, weights, self.reg_strength, N),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-9, "gtol": 1e-6},
        )
        if not np.all(np.isfinite(result.x)):
            raise DataValidationError(
                f"TeamStrengthPoissonModel optimisation failed: {result.message}"
            )

        self.mu_ = float(result.x[0])
        self.home_adv_ = float(result.x[1])
        alpha_arr = result.x[2 : 2 + N]
        beta_arr = result.x[2 + N : 2 + 2 * N]
        self.attack_ = dict(zip(teams, alpha_arr.tolist()))
        self.defense_ = dict(zip(teams, beta_arr.tolist()))

        if self.use_dc_correction:
            expected = self.predict_expected_goals(training_features)
            self.rho_ = optimize_rho(
                expected_home_goals=expected["expected_home_goals"].to_numpy(dtype=float),
                expected_away_goals=expected["expected_away_goals"].to_numpy(dtype=float),
                actual_home_goals=home_scores,
                actual_away_goals=away_scores,
                max_goals=self.max_goals,
                rho_bounds=self.rho_bounds,
                sample_weight=weights,
            )
        else:
            self.rho_ = 0.0

        return self

    def predict_expected_goals(self, features: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        _require_cols(features, _REQUIRED_PRED_COLS)

        home_alpha = np.array([self.attack_.get(str(t), 0.0) for t in features["home_team"]])
        away_alpha = np.array([self.attack_.get(str(t), 0.0) for t in features["away_team"]])
        home_beta = np.array([self.defense_.get(str(t), 0.0) for t in features["home_team"]])
        away_beta = np.array([self.defense_.get(str(t), 0.0) for t in features["away_team"]])
        venue_adv = _venue_multiplier(features)

        log_lh = self.mu_ + self.home_adv_ * venue_adv + home_alpha - away_beta
        log_la = self.mu_ - self.home_adv_ * venue_adv + away_alpha - home_beta

        return pd.DataFrame(
            {
                "expected_home_goals": np.clip(
                    np.exp(log_lh), self.min_expected_goals, self.max_expected_goals
                ),
                "expected_away_goals": np.clip(
                    np.exp(log_la), self.min_expected_goals, self.max_expected_goals
                ),
            },
            index=features.index,
        )

    def predict_score_matrix(self, features: pd.DataFrame | pd.Series) -> np.ndarray:
        self._check_is_fitted()
        frame = features.to_frame().T if isinstance(features, pd.Series) else features
        exp = self.predict_expected_goals(frame).iloc[0]
        return self._matrix(float(exp["expected_home_goals"]), float(exp["expected_away_goals"]))

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        rows: list[dict[str, Any]] = []
        for _, exp in self.predict_expected_goals(features).iterrows():
            lh, la = float(exp["expected_home_goals"]), float(exp["expected_away_goals"])
            matrix = self._matrix(lh, la)
            scoreline_options = top_scorelines(matrix, count=5)
            most_likely = scoreline_options[0]
            rows.append(
                {
                    "expected_home_goals": lh,
                    "expected_away_goals": la,
                    "pred_home_goals": most_likely["home_goals"],
                    "pred_away_goals": most_likely["away_goals"],
                    "most_likely_score_prob": most_likely["probability"],
                    "top_scorelines": scoreline_options,
                    **outcome_probabilities(matrix),
                }
            )
        return pd.DataFrame(rows, index=features.index)

    def _matrix(self, lh: float, la: float) -> np.ndarray:
        if self.use_dc_correction and self.rho_ != 0.0:
            return dixon_coles_score_matrix_from_expected_goals(
                lh, la, rho=self.rho_, max_goals=self.max_goals
            )
        return score_matrix_from_expected_goals(lh, la, max_goals=self.max_goals)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "mu_"):
            raise DataValidationError("TeamStrengthPoissonModel must be fitted before prediction")


def _venue_multiplier(features: pd.DataFrame) -> np.ndarray:
    return (
        features["home_host_advantage"].fillna(False).astype(float).to_numpy()
        - features["away_host_advantage"].fillna(False).astype(float).to_numpy()
    )


def _nll_and_grad(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_scores: np.ndarray,
    away_scores: np.ndarray,
    venue_adv: np.ndarray,
    weights: np.ndarray,
    reg_strength: float,
    N: int,
) -> tuple[float, np.ndarray]:
    mu = params[0]
    h = params[1]
    alpha = params[2 : 2 + N]
    beta = params[2 + N : 2 + 2 * N]

    log_lh = mu + h * venue_adv + alpha[home_idx] - beta[away_idx]
    log_la = mu - h * venue_adv + alpha[away_idx] - beta[home_idx]
    lh = np.exp(log_lh)
    la = np.exp(log_la)

    nll = float(
        np.sum(weights * (lh - home_scores * log_lh + la - away_scores * log_la))
        + reg_strength * (float(np.dot(alpha, alpha)) + float(np.dot(beta, beta)))
    )

    # Weighted Pearson residuals (positive = over-predicted)
    r_h = weights * (lh - home_scores)
    r_a = weights * (la - away_scores)

    grad_mu = float(np.sum(r_h + r_a))
    grad_h = float(np.sum(venue_adv * (r_h - r_a)))

    g_alpha = np.zeros(N)
    np.add.at(g_alpha, home_idx, r_h)
    np.add.at(g_alpha, away_idx, r_a)
    g_alpha += 2.0 * reg_strength * alpha

    g_beta = np.zeros(N)
    np.add.at(g_beta, away_idx, -r_h)
    np.add.at(g_beta, home_idx, -r_a)
    g_beta += 2.0 * reg_strength * beta

    grad = np.empty(2 + 2 * N)
    grad[0] = grad_mu
    grad[1] = grad_h
    grad[2 : 2 + N] = g_alpha
    grad[2 + N : 2 + 2 * N] = g_beta
    return nll, grad


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise DataValidationError(f"TeamStrengthPoissonModel: missing columns: {', '.join(missing)}")

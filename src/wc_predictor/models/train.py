from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from wc_predictor.data import DataValidationError
from wc_predictor.models.dixon_coles import DixonColesScoreModel
from wc_predictor.models.ensemble import EnsembleScoreModel
from wc_predictor.models.poisson_baseline import SimplePoissonScoreModel
from wc_predictor.models.team_strength import TeamStrengthPoissonModel

_DEFAULT_ALPHA_CANDIDATES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
_DEFAULT_ENSEMBLE_WEIGHT_CANDIDATES = ((0.25, 0.75), (0.5, 0.5), (0.75, 0.25))
_TEAM_STRENGTH_MODEL_NAMES = {"team_strength", "team-strength"}
_ENSEMBLE_MODEL_NAMES = {"ensemble"}

_FACTORIAL_LUT = np.array([math.factorial(k) for k in range(20)], dtype=float)


@dataclass(frozen=True)
class TunedModelHyperparameters:
    alpha: float
    reg_strength: float | None = None
    model_weights: tuple[float, ...] | None = None


def train_poisson_baseline(
    training_features: pd.DataFrame,
    max_goals: int = 8,
    alpha: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> SimplePoissonScoreModel:
    model = SimplePoissonScoreModel(max_goals=max_goals, alpha=alpha)
    return model.fit(training_features, sample_weight=sample_weight)


def train_dixon_coles(
    training_features: pd.DataFrame,
    max_goals: int = 8,
    alpha: float = 1.0,
    rho_bounds: tuple[float, float] = (-0.3, 0.3),
    sample_weight: np.ndarray | None = None,
) -> DixonColesScoreModel:
    model = DixonColesScoreModel(max_goals=max_goals, alpha=alpha, rho_bounds=rho_bounds)
    return model.fit(training_features, sample_weight=sample_weight)


def train_team_strength(
    training_features: pd.DataFrame,
    max_goals: int = 8,
    reg_strength: float = 0.01,
    rho_bounds: tuple[float, float] = (-0.3, 0.3),
    use_dc_correction: bool = True,
    sample_weight: np.ndarray | None = None,
) -> TeamStrengthPoissonModel:
    model = TeamStrengthPoissonModel(
        max_goals=max_goals,
        reg_strength=reg_strength,
        rho_bounds=rho_bounds,
        use_dc_correction=use_dc_correction,
    )
    return model.fit(training_features, sample_weight=sample_weight)


def train_ensemble(
    training_features: pd.DataFrame,
    max_goals: int = 8,
    alpha: float = 1.0,
    reg_strength: float = 0.01,
    rho_bounds: tuple[float, float] = (-0.3, 0.3),
    model_weights: list[float] | None = None,
    sample_weight: np.ndarray | None = None,
) -> EnsembleScoreModel:
    poisson = train_poisson_baseline(
        training_features, max_goals=max_goals, alpha=alpha, sample_weight=sample_weight
    )
    ts = train_team_strength(
        training_features,
        max_goals=max_goals,
        reg_strength=reg_strength,
        rho_bounds=rho_bounds,
        use_dc_correction=True,
        sample_weight=sample_weight,
    )
    return EnsembleScoreModel(
        models=[poisson, ts],
        weights=model_weights,
        max_goals=max_goals,
    )


def train_score_model(
    training_features: pd.DataFrame,
    model_type: str = "poisson",
    max_goals: int = 8,
    alpha: float = 1.0,
    rho_bounds: tuple[float, float] = (-0.3, 0.3),
    sample_weight: np.ndarray | None = None,
    reg_strength: float | None = None,
    model_weights: list[float] | tuple[float, ...] | None = None,
) -> SimplePoissonScoreModel | DixonColesScoreModel | TeamStrengthPoissonModel | EnsembleScoreModel:
    normalized = model_type.lower().strip()
    if normalized in {"poisson", "simple_poisson", "poisson_baseline"}:
        return train_poisson_baseline(
            training_features, max_goals=max_goals, alpha=alpha, sample_weight=sample_weight
        )
    if normalized in {"dixon_coles", "dixon-coles"}:
        return train_dixon_coles(
            training_features,
            max_goals=max_goals,
            alpha=alpha,
            rho_bounds=rho_bounds,
            sample_weight=sample_weight,
        )
    if normalized in _TEAM_STRENGTH_MODEL_NAMES:
        return train_team_strength(
            training_features,
            max_goals=max_goals,
            reg_strength=alpha if reg_strength is None else reg_strength,
            rho_bounds=rho_bounds,
            sample_weight=sample_weight,
        )
    if normalized in _ENSEMBLE_MODEL_NAMES:
        return train_ensemble(
            training_features,
            max_goals=max_goals,
            alpha=alpha,
            reg_strength=alpha if reg_strength is None else reg_strength,
            rho_bounds=rho_bounds,
            model_weights=list(model_weights) if model_weights is not None else None,
            sample_weight=sample_weight,
        )
    raise DataValidationError(f"unknown model type: {model_type}")


def _batch_score_matrices(
    expected_home: np.ndarray,
    expected_away: np.ndarray,
    max_goals: int,
) -> np.ndarray:
    """Return (n, max_goals+1, max_goals+1) Poisson score-probability matrices, vectorised."""
    k = max_goals + 1
    g = np.arange(k, dtype=float)
    fact = _FACTORIAL_LUT[:k]
    lh = expected_home[:, None]
    la = expected_away[:, None]
    hp = np.exp(-lh) * (lh ** g) / fact
    ap = np.exp(-la) * (la ** g) / fact
    mats = hp[:, :, None] * ap[:, None, :]
    totals = mats.sum(axis=(1, 2), keepdims=True)
    return np.where(totals > 0, mats / totals, mats)


def _rps_from_matrices(
    matrices: np.ndarray,
    home_scores: np.ndarray,
    away_scores: np.ndarray,
) -> float:
    """Mean RPS from a batch of (max_goals+1, max_goals+1) score matrices."""
    k = matrices.shape[1]
    tril_i, tril_j = np.tril_indices(k, k=-1)
    diag = np.arange(k)
    prob_home_win = matrices[:, tril_i, tril_j].sum(axis=1)
    prob_draw = matrices[:, diag, diag].sum(axis=1)
    c1 = (home_scores > away_scores).astype(float)
    c2 = (home_scores >= away_scores).astype(float)
    rps = 0.5 * ((prob_home_win - c1) ** 2 + (prob_home_win + prob_draw - c2) ** 2)
    return float(rps.mean()) if len(rps) > 0 else float("inf")


def tune_alpha(
    training_features: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
    candidates: tuple[float, ...] = _DEFAULT_ALPHA_CANDIDATES,
    n_splits: int = 5,
    max_goals: int = 8,
    model_type: str = "poisson",
) -> float:
    """Return the regularisation strength that minimises mean RPS via time-series CV."""
    return tune_model_hyperparameters(
        training_features=training_features,
        sample_weight=sample_weight,
        candidates=candidates,
        n_splits=n_splits,
        max_goals=max_goals,
        model_type=model_type,
    ).alpha


def tune_model_hyperparameters(
    training_features: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
    candidates: tuple[float, ...] = _DEFAULT_ALPHA_CANDIDATES,
    n_splits: int = 5,
    max_goals: int = 8,
    model_type: str = "poisson",
    ensemble_weight_candidates: tuple[tuple[float, float], ...] = _DEFAULT_ENSEMBLE_WEIGHT_CANDIDATES,
) -> TunedModelHyperparameters:
    """Tune model hyperparameters using chronological CV and RPS."""
    normalized = model_type.lower().strip()
    sorted_df = training_features.sort_values("date").reset_index(drop=True)
    sorted_w = _sort_weights_like_features(training_features, sample_weight)

    if normalized in _ENSEMBLE_MODEL_NAMES:
        return _tune_ensemble(
            sorted_df=sorted_df,
            sorted_w=sorted_w,
            candidates=candidates,
            n_splits=n_splits,
            max_goals=max_goals,
            ensemble_weight_candidates=ensemble_weight_candidates,
        )

    is_ts = normalized in _TEAM_STRENGTH_MODEL_NAMES
    max_splits = 3 if is_ts else n_splits
    effective_splits = min(max_splits, len(sorted_df) - 1)
    if effective_splits < 2:
        return TunedModelHyperparameters(alpha=float(candidates[0]))

    tscv = TimeSeriesSplit(n_splits=effective_splits)
    best_alpha = candidates[0]
    best_loss = float("inf")

    for alpha in candidates:
        fold_losses: list[float] = []
        for train_idx, val_idx in tscv.split(sorted_df):
            train_df = sorted_df.iloc[train_idx]
            val_df = sorted_df.iloc[val_idx]
            w = sorted_w[train_idx] if sorted_w is not None else None
            try:
                model = _tuning_model(normalized, max_goals=max_goals, alpha=alpha)
                model.fit(train_df, sample_weight=w)
                eg = model.predict_expected_goals(val_df)
                mats = _batch_score_matrices(
                    eg["expected_home_goals"].to_numpy(dtype=float),
                    eg["expected_away_goals"].to_numpy(dtype=float),
                    max_goals,
                )
                loss = _rps_from_matrices(
                    mats,
                    val_df["home_score"].to_numpy(dtype=float),
                    val_df["away_score"].to_numpy(dtype=float),
                )
            except Exception:
                loss = float("inf")
            fold_losses.append(loss)

        mean_loss = float(np.mean(fold_losses)) if fold_losses else float("inf")
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_alpha = alpha

    reg_strength = float(best_alpha) if is_ts else None
    return TunedModelHyperparameters(alpha=float(best_alpha), reg_strength=reg_strength)


def _tune_ensemble(
    sorted_df: pd.DataFrame,
    sorted_w: np.ndarray | None,
    candidates: tuple[float, ...],
    n_splits: int,
    max_goals: int,
    ensemble_weight_candidates: tuple[tuple[float, float], ...],
) -> TunedModelHyperparameters:
    effective_splits = min(3, n_splits, len(sorted_df) - 1)
    if effective_splits < 2:
        return TunedModelHyperparameters(
            alpha=float(candidates[0]),
            reg_strength=float(candidates[0]),
            model_weights=tuple(ensemble_weight_candidates[0]),
        )

    tscv = TimeSeriesSplit(n_splits=effective_splits)
    best = TunedModelHyperparameters(
        alpha=float(candidates[0]),
        reg_strength=float(candidates[0]),
        model_weights=tuple(ensemble_weight_candidates[0]),
    )
    best_loss = float("inf")

    for alpha in candidates:
        losses_by_weight = {tuple(weights): [] for weights in ensemble_weight_candidates}
        for train_idx, val_idx in tscv.split(sorted_df):
            train_df = sorted_df.iloc[train_idx]
            val_df = sorted_df.iloc[val_idx]
            w = sorted_w[train_idx] if sorted_w is not None else None
            try:
                poisson = SimplePoissonScoreModel(
                    max_goals=max_goals, alpha=alpha, max_iter=200
                ).fit(train_df, sample_weight=w)
                team_strength = TeamStrengthPoissonModel(
                    max_goals=max_goals,
                    reg_strength=alpha,
                    use_dc_correction=False,
                    max_iter=200,
                ).fit(train_df, sample_weight=w)
                eg_p = poisson.predict_expected_goals(val_df)
                eg_ts = team_strength.predict_expected_goals(val_df)
                mats_p = _batch_score_matrices(
                    eg_p["expected_home_goals"].to_numpy(dtype=float),
                    eg_p["expected_away_goals"].to_numpy(dtype=float),
                    max_goals,
                )
                mats_ts = _batch_score_matrices(
                    eg_ts["expected_home_goals"].to_numpy(dtype=float),
                    eg_ts["expected_away_goals"].to_numpy(dtype=float),
                    max_goals,
                )
                h_val = val_df["home_score"].to_numpy(dtype=float)
                a_val = val_df["away_score"].to_numpy(dtype=float)
                for weights in ensemble_weight_candidates:
                    w0, w1 = float(weights[0]), float(weights[1])
                    blended = w0 * mats_p + w1 * mats_ts
                    totals = blended.sum(axis=(1, 2), keepdims=True)
                    blended = np.where(totals > 0, blended / totals, blended)
                    losses_by_weight[tuple(weights)].append(
                        _rps_from_matrices(blended, h_val, a_val)
                    )
            except Exception:
                for weights in ensemble_weight_candidates:
                    losses_by_weight[tuple(weights)].append(float("inf"))

        for weights, losses in losses_by_weight.items():
            mean_loss = float(np.mean(losses)) if losses else float("inf")
            if mean_loss < best_loss:
                best_loss = mean_loss
                best = TunedModelHyperparameters(
                    alpha=float(alpha),
                    reg_strength=float(alpha),
                    model_weights=tuple(float(weight) for weight in weights),
                )

    return best


def _tuning_model(
    normalized_model_type: str,
    max_goals: int,
    alpha: float,
) -> SimplePoissonScoreModel | TeamStrengthPoissonModel:
    if normalized_model_type in _TEAM_STRENGTH_MODEL_NAMES:
        return TeamStrengthPoissonModel(
            max_goals=max_goals,
            reg_strength=alpha,
            use_dc_correction=False,
            max_iter=200,
        )
    return SimplePoissonScoreModel(max_goals=max_goals, alpha=alpha, max_iter=200)


def _sort_weights_like_features(
    training_features: pd.DataFrame,
    sample_weight: np.ndarray | None,
) -> np.ndarray | None:
    if sample_weight is None:
        return None
    sorted_index = training_features.sort_values("date").index
    return pd.Series(sample_weight, index=training_features.index).loc[
        sorted_index
    ].to_numpy(dtype=float)



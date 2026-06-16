"""Model training and prediction helpers."""

from wc_predictor.models.dixon_coles import (
    DixonColesScoreModel,
    dixon_coles_score_matrix_from_expected_goals,
    dixon_coles_tau,
    optimize_rho,
)
from wc_predictor.models.ensemble import EnsembleScoreModel
from wc_predictor.models.poisson_baseline import (
    DEFAULT_FEATURE_COLUMNS,
    SimplePoissonScoreModel,
    outcome_probabilities,
    score_matrix_from_expected_goals,
    top_scorelines,
)
from wc_predictor.models.predict import MatchPredictor, predict_match
from wc_predictor.models.team_strength import TeamStrengthPoissonModel
from wc_predictor.models.train import (
    TunedModelHyperparameters,
    train_dixon_coles,
    train_ensemble,
    train_poisson_baseline,
    train_score_model,
    train_team_strength,
    tune_model_hyperparameters,
)

__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "DixonColesScoreModel",
    "EnsembleScoreModel",
    "MatchPredictor",
    "SimplePoissonScoreModel",
    "TeamStrengthPoissonModel",
    "TunedModelHyperparameters",
    "dixon_coles_score_matrix_from_expected_goals",
    "dixon_coles_tau",
    "optimize_rho",
    "outcome_probabilities",
    "predict_match",
    "score_matrix_from_expected_goals",
    "top_scorelines",
    "train_dixon_coles",
    "train_ensemble",
    "train_poisson_baseline",
    "train_score_model",
    "train_team_strength",
    "tune_model_hyperparameters",
]

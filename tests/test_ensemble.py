from pathlib import Path

import numpy as np
import pytest

from tests.helpers import make_test_dir
from wc_predictor.data import load_results
from wc_predictor.features import MatchFeatureBuilder
from wc_predictor.models import (
    EnsembleScoreModel,
    SimplePoissonScoreModel,
    TeamStrengthPoissonModel,
    train_ensemble,
    train_score_model,
    tune_model_hyperparameters,
)


def write_file(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def sample_results(tmp_path: Path) -> Path:
    return write_file(
        tmp_path,
        "results.csv",
        """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2020-01-01,Alpha,Beta,2,0,Friendly,A City,Nowhere,true
2020-01-02,Gamma,Delta,1,1,Friendly,B City,Nowhere,true
2020-01-03,Alpha,Gamma,3,1,Friendly,C City,Nowhere,true
2020-01-04,Beta,Delta,0,1,Friendly,D City,Nowhere,true
2020-01-05,Alpha,Delta,2,2,Friendly,E City,Nowhere,true
2020-01-06,Beta,Gamma,1,2,Friendly,F City,Nowhere,true
2020-01-07,Delta,Alpha,0,2,Friendly,G City,Nowhere,true
2020-01-08,Gamma,Beta,1,0,Friendly,H City,Nowhere,true
2020-01-09,Alpha,Beta,1,1,Friendly,I City,Alpha,false
2020-01-10,Delta,Gamma,2,1,Friendly,J City,Nowhere,true
""",
    )


def trained_ensemble(tmp_path: Path | None = None) -> tuple[EnsembleScoreModel, MatchFeatureBuilder]:
    tmp_path = tmp_path or make_test_dir()
    results = load_results(sample_results(tmp_path))
    fb = MatchFeatureBuilder(results)
    training = fb.build_training_features()
    model = train_ensemble(training, max_goals=8, alpha=0.5, reg_strength=0.01)
    return model, fb


def test_ensemble_model_contains_two_sub_models() -> None:
    model, _ = trained_ensemble()
    assert len(model.models) == 2
    assert isinstance(model.models[0], SimplePoissonScoreModel)
    assert isinstance(model.models[1], TeamStrengthPoissonModel)


def test_ensemble_default_weights_sum_to_one() -> None:
    model, _ = trained_ensemble()
    assert sum(model.weights_) == pytest.approx(1.0)
    assert model.weights_[0] == pytest.approx(0.5)
    assert model.weights_[1] == pytest.approx(0.5)


def test_ensemble_custom_weights_are_normalised() -> None:
    tmp = make_test_dir()
    results = load_results(sample_results(tmp))
    fb = MatchFeatureBuilder(results)
    training = fb.build_training_features()
    model = train_ensemble(training, max_goals=8, alpha=0.5, model_weights=[3.0, 1.0])
    assert model.weights_[0] == pytest.approx(0.75)
    assert model.weights_[1] == pytest.approx(0.25)


def test_ensemble_predict_proba_sums_to_one() -> None:
    model, fb = trained_ensemble()
    features = fb.build_match_features("Alpha", "Gamma", prediction_date="2026-06-01")
    pred = model.predict_proba(features).iloc[0]
    total = pred["prob_home_win"] + pred["prob_draw"] + pred["prob_away_win"]
    assert total == pytest.approx(1.0)


def test_ensemble_predict_proba_returns_positive_expected_goals() -> None:
    model, fb = trained_ensemble()
    features = fb.build_match_features("Alpha", "Beta", prediction_date="2026-06-01")
    pred = model.predict_proba(features).iloc[0]
    assert pred["expected_home_goals"] > 0
    assert pred["expected_away_goals"] > 0


def test_ensemble_predict_score_matrix_sums_to_one() -> None:
    model, fb = trained_ensemble()
    features = fb.build_match_features("Alpha", "Delta", prediction_date="2026-06-01")
    matrix = model.predict_score_matrix(features)
    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()


def test_ensemble_predict_score_matrix_from_series() -> None:
    model, fb = trained_ensemble()
    features = fb.build_match_features("Beta", "Gamma", prediction_date="2026-06-01")
    matrix = model.predict_score_matrix(features.iloc[0])
    assert matrix.sum() == pytest.approx(1.0)


def test_ensemble_predict_expected_goals_batch() -> None:
    model, fb = trained_ensemble()
    features = fb.build_match_features("Alpha", "Gamma", prediction_date="2026-06-01")
    eg = model.predict_expected_goals(features)
    assert list(eg.columns) == ["expected_home_goals", "expected_away_goals"]
    assert (eg["expected_home_goals"] > 0).all()
    assert (eg["expected_away_goals"] > 0).all()


def test_ensemble_matrix_is_between_sub_model_matrices() -> None:
    """Averaged matrix should be component-wise between the two sub-model matrices."""
    model, fb = trained_ensemble()
    features = fb.build_match_features("Alpha", "Beta", prediction_date="2026-06-01")
    row = features.iloc[[0]].reset_index(drop=True)

    m0 = model.models[0].predict_score_matrix(row)
    m1 = model.models[1].predict_score_matrix(row)
    me = model.predict_score_matrix(row)

    sz = me.shape[0]
    low = np.minimum(m0[:sz, :sz], m1[:sz, :sz])
    high = np.maximum(m0[:sz, :sz], m1[:sz, :sz])
    assert (me >= low - 1e-9).all()
    assert (me <= high + 1e-9).all()


def test_train_score_model_dispatches_ensemble() -> None:
    tmp = make_test_dir()
    results = load_results(sample_results(tmp))
    fb = MatchFeatureBuilder(results)
    training = fb.build_training_features()
    model = train_score_model(training, model_type="ensemble", max_goals=8, alpha=0.5)
    assert isinstance(model, EnsembleScoreModel)


def test_tune_model_hyperparameters_tunes_ensemble_weights() -> None:
    tmp = make_test_dir()
    results = load_results(sample_results(tmp))
    fb = MatchFeatureBuilder(results)
    training = fb.build_training_features()

    tuned = tune_model_hyperparameters(
        training,
        candidates=(0.01, 0.1),
        n_splits=2,
        max_goals=8,
        model_type="ensemble",
        ensemble_weight_candidates=((0.25, 0.75), (0.75, 0.25)),
    )

    assert tuned.alpha in {0.01, 0.1}
    assert tuned.reg_strength == tuned.alpha
    assert tuned.model_weights in {(0.25, 0.75), (0.75, 0.25)}


def test_ensemble_raises_on_empty_model_list() -> None:
    with pytest.raises(Exception):
        EnsembleScoreModel(models=[], max_goals=8)


def test_ensemble_raises_on_weight_length_mismatch() -> None:
    tmp = make_test_dir()
    results = load_results(sample_results(tmp))
    fb = MatchFeatureBuilder(results)
    training = fb.build_training_features()
    model = train_ensemble(training, max_goals=8, alpha=0.5)
    with pytest.raises(Exception):
        EnsembleScoreModel(models=model.models, weights=[0.5, 0.3, 0.2])

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import make_test_dir
from wc_predictor.data import load_results
from wc_predictor.features import MatchFeatureBuilder
from wc_predictor.models import (
    DixonColesScoreModel,
    dixon_coles_score_matrix_from_expected_goals,
    dixon_coles_tau,
    score_matrix_from_expected_goals,
    train_dixon_coles,
    train_score_model,
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


def test_dixon_coles_tau_adjusts_only_low_scores() -> None:
    assert dixon_coles_tau(0, 0, 1.2, 1.1, 0.1) == pytest.approx(0.868)
    assert dixon_coles_tau(0, 1, 1.2, 1.1, 0.1) == pytest.approx(1.12)
    assert dixon_coles_tau(1, 0, 1.2, 1.1, 0.1) == pytest.approx(1.11)
    assert dixon_coles_tau(1, 1, 1.2, 1.1, 0.1) == pytest.approx(0.9)
    assert dixon_coles_tau(2, 1, 1.2, 1.1, 0.1) == pytest.approx(1.0)


def test_dixon_coles_score_matrix_sums_to_one_and_changes_low_scores() -> None:
    base = score_matrix_from_expected_goals(1.2, 1.1, max_goals=8)
    adjusted = dixon_coles_score_matrix_from_expected_goals(1.2, 1.1, rho=0.1, max_goals=8)

    assert adjusted.sum() == pytest.approx(1.0)
    assert (adjusted >= 0).all()
    assert adjusted[0, 0] != pytest.approx(base[0, 0])
    assert adjusted[2, 2] == pytest.approx(base[2, 2] / (base * 1).sum(), rel=0.2)


def test_train_dixon_coles_fits_rho_within_bounds_and_predicts() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    feature_builder = MatchFeatureBuilder(results)
    training_features = feature_builder.build_training_features()
    matchup_features = feature_builder.build_match_features(
        "Alpha",
        "Gamma",
        prediction_date="2026-06-01",
    )

    model = train_dixon_coles(
        training_features,
        max_goals=8,
        alpha=0.5,
        rho_bounds=(-0.2, 0.2),
    )
    prediction = model.predict_proba(matchup_features).iloc[0]
    matrix = model.predict_score_matrix(matchup_features)

    assert isinstance(model, DixonColesScoreModel)
    assert -0.2 <= model.rho_ <= 0.2
    assert prediction["expected_home_goals"] > 0
    assert prediction["prob_home_win"] + prediction["prob_draw"] + prediction["prob_away_win"] == pytest.approx(1.0)
    assert matrix.sum() == pytest.approx(1.0)


def test_train_score_model_dispatches_dixon_coles() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    feature_builder = MatchFeatureBuilder(results)
    training_features = feature_builder.build_training_features()

    model = train_score_model(
        training_features,
        model_type="dixon_coles",
        max_goals=8,
        alpha=0.5,
        rho_bounds=(-0.2, 0.2),
    )

    assert isinstance(model, DixonColesScoreModel)


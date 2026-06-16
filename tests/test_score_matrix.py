from pathlib import Path

import numpy as np
import pytest

from tests.helpers import make_test_dir
from wc_predictor.data import DataValidationError, load_results
from wc_predictor.features import MatchFeatureBuilder
from wc_predictor.models import (
    SimplePoissonScoreModel,
    outcome_probabilities,
    score_matrix_from_expected_goals,
    top_scorelines,
    train_poisson_baseline,
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


def test_score_matrix_sums_to_one_and_is_non_negative() -> None:
    matrix = score_matrix_from_expected_goals(1.4, 1.1, max_goals=8)

    assert matrix.shape == (9, 9)
    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()


def test_score_matrix_rejects_non_positive_expected_goals() -> None:
    with pytest.raises(DataValidationError, match="expected goals must be positive"):
        score_matrix_from_expected_goals(0, 1.1)


def test_outcome_probabilities_sum_to_one() -> None:
    matrix = score_matrix_from_expected_goals(1.4, 1.1, max_goals=8)

    probabilities = outcome_probabilities(matrix)

    total = sum(probabilities.values())
    assert total == pytest.approx(1.0)
    assert set(probabilities) == {"prob_home_win", "prob_draw", "prob_away_win"}


def test_top_scoreline_matches_matrix_maximum() -> None:
    matrix = score_matrix_from_expected_goals(1.8, 0.9, max_goals=8)

    top = top_scorelines(matrix, count=5)
    first = top[0]

    assert len(top) == 5
    assert first["probability"] == pytest.approx(matrix.max())
    assert matrix[first["home_goals"], first["away_goals"]] == pytest.approx(matrix.max())
    assert [item["probability"] for item in top] == sorted(
        [item["probability"] for item in top], reverse=True
    )


def test_simple_poisson_model_predicts_positive_expected_goals() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    feature_builder = MatchFeatureBuilder(results)
    training_features = feature_builder.build_training_features()
    matchup_features = feature_builder.build_match_features(
        "Alpha",
        "Gamma",
        venue_mode="neutral",
        prediction_date="2026-06-01",
    )

    model = SimplePoissonScoreModel(max_goals=8, alpha=0.5).fit(training_features)
    expected = model.predict_expected_goals(matchup_features)

    assert expected.loc[0, "expected_home_goals"] > 0
    assert expected.loc[0, "expected_away_goals"] > 0


def test_simple_poisson_model_predicts_normalized_score_matrix() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    feature_builder = MatchFeatureBuilder(results)
    training_features = feature_builder.build_training_features()
    matchup_features = feature_builder.build_match_features(
        "Alpha",
        "Gamma",
        venue_mode="neutral",
        prediction_date="2026-06-01",
    )

    model = SimplePoissonScoreModel(max_goals=8, alpha=0.5).fit(training_features)
    matrix = model.predict_score_matrix(matchup_features)

    assert matrix.sum() == pytest.approx(1.0)
    assert np.isfinite(matrix).all()
    assert (matrix >= 0).all()


def test_simple_poisson_model_predict_proba_returns_score_outputs() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    feature_builder = MatchFeatureBuilder(results)
    training_features = feature_builder.build_training_features()
    matchup_features = feature_builder.build_match_features(
        "Alpha",
        "Gamma",
        venue_mode="neutral",
        prediction_date="2026-06-01",
    )

    model = SimplePoissonScoreModel(max_goals=8, alpha=0.5).fit(training_features)
    predictions = model.predict_proba(matchup_features)
    row = predictions.iloc[0]

    assert row["expected_home_goals"] > 0
    assert row["expected_away_goals"] > 0
    assert row["prob_home_win"] + row["prob_draw"] + row["prob_away_win"] == pytest.approx(1.0)
    assert isinstance(row["top_scorelines"], list)
    assert len(row["top_scorelines"]) == 5
    assert row["most_likely_score_prob"] == pytest.approx(
        row["top_scorelines"][0]["probability"]
    )


def test_simple_poisson_model_rejects_missing_feature_values() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    feature_builder = MatchFeatureBuilder(results)
    training_features = feature_builder.build_training_features()
    matchup_features = feature_builder.build_match_features(
        "Alpha",
        "Gamma",
        venue_mode="neutral",
        prediction_date="2026-06-01",
    )
    matchup_features.loc[0, "home_elo_pre"] = np.nan

    model = SimplePoissonScoreModel(max_goals=8, alpha=0.5).fit(training_features)

    with pytest.raises(DataValidationError, match="home_elo_pre"):
        model.predict_expected_goals(matchup_features)


def test_train_poisson_baseline_returns_fitted_model() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    feature_builder = MatchFeatureBuilder(results)
    training_features = feature_builder.build_training_features()
    matchup_features = feature_builder.build_match_features(
        "Alpha",
        "Gamma",
        venue_mode="neutral",
        prediction_date="2026-06-01",
    )

    model = train_poisson_baseline(training_features, max_goals=8, alpha=0.5)
    predictions = model.predict_proba(matchup_features)

    assert predictions.loc[0, "expected_home_goals"] > 0

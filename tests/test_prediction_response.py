from pathlib import Path

import pytest

from tests.helpers import make_test_dir
from wc_predictor.data import DataValidationError, load_results
from wc_predictor.features import MatchFeatureBuilder
from wc_predictor.models import MatchPredictor, predict_match, train_poisson_baseline
from wc_predictor.presentation import (
    ConfidenceThresholds,
    confidence_explanation,
    confidence_label,
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


def build_predictor() -> MatchPredictor:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    feature_builder = MatchFeatureBuilder(results)
    model = train_poisson_baseline(
        feature_builder.build_training_features(),
        max_goals=8,
        alpha=0.5,
    )
    return MatchPredictor(
        model=model,
        feature_builder=feature_builder,
        model_version="test-model",
        feature_cutoff_date="2026-06-01",
    )


def test_confidence_label_follows_thresholds() -> None:
    thresholds = ConfidenceThresholds(low_max=0.12, medium_max=0.18)

    assert confidence_label(0.119, thresholds) == "Low"
    assert confidence_label(0.12, thresholds) == "Medium"
    assert confidence_label(0.18, thresholds) == "Medium"
    assert confidence_label(0.181, thresholds) == "High"


def test_confidence_explanation_is_plain_language() -> None:
    assert "scoreline" in confidence_explanation("Low")
    assert "plausible" in confidence_explanation("Medium")
    assert "strong" in confidence_explanation("High")


def test_predict_match_response_includes_required_ui_fields() -> None:
    predictor = build_predictor()

    response = predictor.predict_match(
        "Alpha",
        "Gamma",
        venue_mode="neutral",
        prediction_date="2026-06-01",
    )

    required_fields = {
        "home_team",
        "away_team",
        "pred_home_goals",
        "pred_away_goals",
        "most_likely_score",
        "most_likely_score_prob",
        "confidence_label",
        "confidence_explanation",
        "top_scorelines",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "expected_home_goals",
        "expected_away_goals",
        "model_version",
        "feature_cutoff_date",
        "low_history_flags",
    }

    assert required_fields.issubset(response)
    assert response["home_team"] == "Alpha"
    assert response["away_team"] == "Gamma"
    assert response["model_version"] == "test-model"
    assert response["feature_cutoff_date"] == "2026-06-01"
    assert response["most_likely_score"].startswith("Alpha ")
    assert response["expected_home_goals"] > 0
    assert response["expected_away_goals"] > 0


def test_predict_match_top_scorelines_are_sorted_and_mark_main_prediction() -> None:
    predictor = build_predictor()

    response = predictor.predict_match("Alpha", "Gamma", prediction_date="2026-06-01")
    top_scorelines = response["top_scorelines"]
    probabilities = [item["probability"] for item in top_scorelines]

    assert len(top_scorelines) == 5
    assert probabilities == sorted(probabilities, reverse=True)
    assert top_scorelines[0]["is_main_prediction"] is True
    assert all(item["rank"] == index + 1 for index, item in enumerate(top_scorelines))


def test_predict_match_outcome_probabilities_sum_to_one() -> None:
    predictor = build_predictor()

    response = predictor.predict_match("Alpha", "Gamma", prediction_date="2026-06-01")
    total = response["prob_home_win"] + response["prob_draw"] + response["prob_away_win"]

    assert total == pytest.approx(1.0)


def test_predict_match_function_wrapper_matches_service_shape() -> None:
    predictor = build_predictor()

    response = predict_match(
        "Alpha",
        "Gamma",
        model=predictor.model,
        feature_builder=predictor.feature_builder,
        prediction_date="2026-06-01",
        model_version="wrapper-model",
        feature_cutoff_date="2026-06-01",
    )

    assert response["model_version"] == "wrapper-model"
    assert response["home_team"] == "Alpha"


def test_predict_match_rejects_same_team() -> None:
    predictor = build_predictor()

    with pytest.raises(DataValidationError, match="must be different"):
        predictor.predict_match("Alpha", "Alpha")


def test_predict_match_rejects_unknown_team() -> None:
    predictor = build_predictor()

    with pytest.raises(DataValidationError, match="unknown team: Atlantis"):
        predictor.predict_match("Alpha", "Atlantis")


def test_predict_match_rejects_invalid_venue_mode() -> None:
    predictor = build_predictor()

    with pytest.raises(DataValidationError, match="invalid venue mode"):
        predictor.predict_match("Alpha", "Gamma", venue_mode="Moon")


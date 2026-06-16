from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_test_dir
from wc_predictor.data import load_results
from wc_predictor.evaluation import (
    BacktestConfig,
    actual_outcome,
    build_evaluation_report,
    evaluate_prediction_frame,
    run_model_comparison_backtests,
    run_world_cup_backtest,
    score_probability,
    write_evaluation_report,
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
2022-06-01,Alpha,Beta,1,0,FIFA World Cup,Final City,Nowhere,true
2022-06-02,Gamma,Delta,0,0,FIFA World Cup,Final City,Nowhere,true
""",
    )


def test_actual_outcome() -> None:
    assert actual_outcome(2, 1) == "H"
    assert actual_outcome(1, 1) == "D"
    assert actual_outcome(0, 1) == "A"


def test_score_probability_handles_in_range_and_out_of_range_scores() -> None:
    matrix = np.ones((3, 3)) / 9

    assert score_probability(matrix, 1, 1) == pytest.approx(1 / 9)
    assert score_probability(matrix, 9, 9) > 0
    assert score_probability(matrix, 9, 9) < 1e-10


def test_evaluate_prediction_frame_calculates_metrics() -> None:
    frame = pd.DataFrame(
        [
            {
                "actual_home_goals": 1,
                "actual_away_goals": 0,
                "expected_home_goals": 1.2,
                "expected_away_goals": 0.8,
                "pred_home_goals": 1,
                "pred_away_goals": 0,
                "top_scorelines": [{"score": "1-0"}, {"score": "1-1"}],
                "actual_score_prob": 0.2,
                "actual_outcome": "H",
                "prob_home_win": 0.5,
                "prob_draw": 0.3,
                "prob_away_win": 0.2,
            },
            {
                "actual_home_goals": 0,
                "actual_away_goals": 0,
                "expected_home_goals": 0.7,
                "expected_away_goals": 0.9,
                "pred_home_goals": 1,
                "pred_away_goals": 1,
                "top_scorelines": [{"score": "1-1"}, {"score": "0-0"}],
                "actual_score_prob": 0.1,
                "actual_outcome": "D",
                "prob_home_win": 0.3,
                "prob_draw": 0.4,
                "prob_away_win": 0.3,
            },
        ]
    )

    metrics = evaluate_prediction_frame(frame)

    assert metrics["exact_score_accuracy"] == pytest.approx(0.5)
    assert metrics["top_5_score_accuracy"] == pytest.approx(1.0)
    assert metrics["home_goals_mae"] > 0
    assert metrics["one_x_two_log_loss"] > 0


def test_run_world_cup_backtest_predicts_sample_tournament() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))

    result = run_world_cup_backtest(
        results,
        2022,
        BacktestConfig(training_start_date="2020-01-01", max_goals=8, alpha=0.5),
    )

    assert result.tournament_year == 2022
    assert result.test_matches == 2
    assert result.predicted_matches == 2
    assert result.skipped_matches == 0
    assert "exact_score_log_loss" in result.metrics


def test_build_and_write_evaluation_report() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    backtest_result = run_world_cup_backtest(
        results,
        2022,
        BacktestConfig(training_start_date="2020-01-01", max_goals=8, alpha=0.5),
    )

    report = build_evaluation_report([backtest_result])
    report_path = tmp_path / "evaluation_report.md"
    write_evaluation_report(report_path, [backtest_result])

    assert "Selected Model" in report
    assert "Backtest Summary" in report
    assert "| 2022 |" in report
    assert report_path.exists()


def test_run_model_comparison_backtests_includes_both_models() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))

    backtest_results = run_model_comparison_backtests(
        results,
        years=[2022],
        model_types=["poisson", "dixon_coles"],
        config=BacktestConfig(training_start_date="2020-01-01", max_goals=8, alpha=0.5),
    )

    assert [result.model_type for result in backtest_results] == ["poisson", "dixon_coles"]
    assert all(result.predicted_matches == 2 for result in backtest_results)

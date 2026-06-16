"""Backtesting and evaluation helpers."""

from wc_predictor.evaluation.backtest import (
    BacktestConfig,
    BacktestResult,
    build_evaluation_report,
    run_project_backtests,
    run_model_comparison_backtests,
    run_world_cup_backtest,
    run_world_cup_backtests,
    write_evaluation_report,
)
from wc_predictor.evaluation.metrics import (
    actual_outcome,
    brier_score,
    draw_calibration,
    evaluate_prediction_frame,
    exact_score_accuracy,
    exact_score_log_loss,
    home_goals_mae,
    one_x_two_log_loss,
    score_probability,
    top_n_score_accuracy,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "actual_outcome",
    "brier_score",
    "build_evaluation_report",
    "draw_calibration",
    "evaluate_prediction_frame",
    "exact_score_accuracy",
    "exact_score_log_loss",
    "home_goals_mae",
    "one_x_two_log_loss",
    "run_project_backtests",
    "run_model_comparison_backtests",
    "run_world_cup_backtest",
    "run_world_cup_backtests",
    "score_probability",
    "top_n_score_accuracy",
    "write_evaluation_report",
]

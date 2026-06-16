from __future__ import annotations

from typing import Any

import pandas as pd


EPSILON = 1e-15


def actual_outcome(home_score: int | float, away_score: int | float) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def exact_score_log_loss(frame: pd.DataFrame) -> float:
    probabilities = frame["actual_score_prob"].clip(lower=EPSILON)
    return float((-probabilities.map(_safe_log)).mean())


def home_goals_mae(frame: pd.DataFrame) -> float:
    return _mae(frame["actual_home_goals"], frame["expected_home_goals"])


def away_goals_mae(frame: pd.DataFrame) -> float:
    return _mae(frame["actual_away_goals"], frame["expected_away_goals"])


def exact_score_accuracy(frame: pd.DataFrame) -> float:
    hits = (
        (frame["actual_home_goals"] == frame["pred_home_goals"])
        & (frame["actual_away_goals"] == frame["pred_away_goals"])
    )
    return float(hits.mean())


def top_n_score_accuracy(frame: pd.DataFrame) -> float:
    hits = []
    for row in frame.itertuples(index=False):
        actual_score = f"{int(row.actual_home_goals)}-{int(row.actual_away_goals)}"
        hits.append(any(item["score"] == actual_score for item in row.top_scorelines))
    return float(pd.Series(hits).mean()) if hits else 0.0


def one_x_two_log_loss(frame: pd.DataFrame) -> float:
    losses = []
    for row in frame.itertuples(index=False):
        probability = _actual_outcome_probability(row)
        losses.append(-_safe_log(max(probability, EPSILON)))
    return float(pd.Series(losses).mean()) if losses else 0.0


def brier_score(frame: pd.DataFrame) -> float:
    scores = []
    for row in frame.itertuples(index=False):
        actual = row.actual_outcome
        scores.append(
            (row.prob_home_win - (1.0 if actual == "H" else 0.0)) ** 2
            + (row.prob_draw - (1.0 if actual == "D" else 0.0)) ** 2
            + (row.prob_away_win - (1.0 if actual == "A" else 0.0)) ** 2
        )
    return float(pd.Series(scores).mean()) if scores else 0.0


def draw_calibration(frame: pd.DataFrame) -> float:
    actual_draw_rate = (frame["actual_outcome"] == "D").mean()
    predicted_draw_rate = frame["prob_draw"].mean()
    return float(predicted_draw_rate - actual_draw_rate)


def ranked_probability_score(frame: pd.DataFrame) -> float:
    """RPS for ordered 3-outcome H/D/A (lower is better)."""
    scores = []
    for row in frame.itertuples(index=False):
        p_h = float(row.prob_home_win)
        p_d = float(row.prob_draw)
        actual = row.actual_outcome
        cdf_pred_1 = p_h
        cdf_pred_2 = p_h + p_d
        cdf_actual_1 = 1.0 if actual == "H" else 0.0
        cdf_actual_2 = 0.0 if actual == "A" else 1.0
        scores.append(0.5 * ((cdf_pred_1 - cdf_actual_1) ** 2 + (cdf_pred_2 - cdf_actual_2) ** 2))
    return float(pd.Series(scores).mean()) if scores else 0.0


def _uniform_rps(frame: pd.DataFrame) -> float:
    """RPS of a 1/3-1/3-1/3 uniform predictor on the same outcomes."""
    scores = []
    for row in frame.itertuples(index=False):
        actual = row.actual_outcome
        cdf_actual_1 = 1.0 if actual == "H" else 0.0
        cdf_actual_2 = 0.0 if actual == "A" else 1.0
        scores.append(0.5 * ((1 / 3 - cdf_actual_1) ** 2 + (2 / 3 - cdf_actual_2) ** 2))
    return float(pd.Series(scores).mean()) if scores else 0.0


def rps_skill_score(frame: pd.DataFrame) -> float:
    """1 - RPS_model / RPS_uniform. Higher is better; 0 = no skill vs uniform."""
    u = _uniform_rps(frame)
    if u <= 0:
        return 0.0
    return 1.0 - ranked_probability_score(frame) / u


def evaluate_prediction_frame(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "exact_score_log_loss": 0.0,
            "home_goals_mae": 0.0,
            "away_goals_mae": 0.0,
            "exact_score_accuracy": 0.0,
            "top_5_score_accuracy": 0.0,
            "one_x_two_log_loss": 0.0,
            "brier_score": 0.0,
            "draw_calibration": 0.0,
            "rps": 0.0,
            "rps_skill_score": 0.0,
        }

    return {
        "exact_score_log_loss": exact_score_log_loss(frame),
        "home_goals_mae": home_goals_mae(frame),
        "away_goals_mae": away_goals_mae(frame),
        "exact_score_accuracy": exact_score_accuracy(frame),
        "top_5_score_accuracy": top_n_score_accuracy(frame),
        "one_x_two_log_loss": one_x_two_log_loss(frame),
        "brier_score": brier_score(frame),
        "draw_calibration": draw_calibration(frame),
        "rps": ranked_probability_score(frame),
        "rps_skill_score": rps_skill_score(frame),
    }


def score_probability(
    matrix: Any,
    home_goals: int | float,
    away_goals: int | float,
) -> float:
    home = int(home_goals)
    away = int(away_goals)
    if home < 0 or away < 0:
        return EPSILON
    if home >= matrix.shape[0] or away >= matrix.shape[1]:
        return EPSILON
    return float(max(matrix[home, away], EPSILON))


def _actual_outcome_probability(row: object) -> float:
    if row.actual_outcome == "H":
        return float(row.prob_home_win)
    if row.actual_outcome == "A":
        return float(row.prob_away_win)
    return float(row.prob_draw)


def _mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float((actual - predicted).abs().mean())


def _safe_log(value: float) -> float:
    import math

    return math.log(max(float(value), EPSILON))


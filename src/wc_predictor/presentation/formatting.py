from __future__ import annotations

from typing import Any

import pandas as pd

from wc_predictor.presentation.confidence import (
    ConfidenceThresholds,
    confidence_hint,
)


def format_prediction_response(
    home_team: str,
    away_team: str,
    prediction_row: pd.Series | dict[str, Any],
    feature_row: pd.Series | dict[str, Any],
    model_version: str,
    feature_cutoff_date: str | pd.Timestamp,
    thresholds: ConfidenceThresholds | None = None,
) -> dict[str, Any]:
    prediction = _to_dict(prediction_row)
    features = _to_dict(feature_row)

    pred_home_goals = int(prediction["pred_home_goals"])
    pred_away_goals = int(prediction["pred_away_goals"])
    probability = float(prediction["most_likely_score_prob"])
    confidence = confidence_hint(probability, thresholds)

    top_scorelines = _format_top_scorelines(prediction["top_scorelines"])

    return {
        "home_team": home_team,
        "away_team": away_team,
        "pred_home_goals": pred_home_goals,
        "pred_away_goals": pred_away_goals,
        "most_likely_score": (
            f"{home_team} {pred_home_goals} - {pred_away_goals} {away_team}"
        ),
        "most_likely_score_prob": probability,
        **confidence,
        "top_scorelines": top_scorelines,
        "prob_home_win": float(prediction["prob_home_win"]),
        "prob_draw": float(prediction["prob_draw"]),
        "prob_away_win": float(prediction["prob_away_win"]),
        "expected_home_goals": float(prediction["expected_home_goals"]),
        "expected_away_goals": float(prediction["expected_away_goals"]),
        "model_version": model_version,
        "feature_cutoff_date": _format_date(feature_cutoff_date),
        "low_history_flags": {
            "home_team": bool(features.get("home_low_history_flag", False)),
            "away_team": bool(features.get("away_low_history_flag", False)),
        },
    }


def _format_top_scorelines(scorelines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for index, scoreline in enumerate(scorelines):
        formatted.append(
            {
                "rank": index + 1,
                "home_goals": int(scoreline["home_goals"]),
                "away_goals": int(scoreline["away_goals"]),
                "score": str(scoreline["score"]),
                "probability": float(scoreline["probability"]),
                "is_main_prediction": index == 0,
            }
        )
    return formatted


def _to_dict(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def _format_date(value: str | pd.Timestamp) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


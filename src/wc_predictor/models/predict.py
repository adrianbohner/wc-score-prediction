from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from wc_predictor.data import DataValidationError
from wc_predictor.features import MatchFeatureBuilder
from wc_predictor.presentation.confidence import ConfidenceThresholds
from wc_predictor.presentation.formatting import format_prediction_response


class ScoreModel(Protocol):
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        ...


@dataclass
class MatchPredictor:
    model: ScoreModel
    feature_builder: MatchFeatureBuilder
    model_version: str = "poisson-baseline-v1"
    feature_cutoff_date: str | pd.Timestamp | None = None
    confidence_thresholds: ConfidenceThresholds | None = None

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        venue_mode: str = "neutral",
        prediction_date: str | pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        home_team = _clean_team_name(home_team, "home_team")
        away_team = _clean_team_name(away_team, "away_team")
        if home_team == away_team:
            raise DataValidationError("home_team and away_team must be different")

        features = self.feature_builder.build_match_features(
            home_team=home_team,
            away_team=away_team,
            venue_mode=venue_mode,
            prediction_date=prediction_date,
        )
        prediction = self.model.predict_proba(features)
        cutoff_date = self.feature_cutoff_date or features.loc[0, "date"]

        return format_prediction_response(
            home_team=home_team,
            away_team=away_team,
            prediction_row=prediction.iloc[0],
            feature_row=features.iloc[0],
            model_version=self.model_version,
            feature_cutoff_date=cutoff_date,
            thresholds=self.confidence_thresholds,
        )


def predict_match(
    home_team: str,
    away_team: str,
    model: ScoreModel,
    feature_builder: MatchFeatureBuilder,
    venue_mode: str = "neutral",
    prediction_date: str | pd.Timestamp | None = None,
    model_version: str = "poisson-baseline-v1",
    feature_cutoff_date: str | pd.Timestamp | None = None,
    confidence_thresholds: ConfidenceThresholds | None = None,
) -> dict[str, Any]:
    predictor = MatchPredictor(
        model=model,
        feature_builder=feature_builder,
        model_version=model_version,
        feature_cutoff_date=feature_cutoff_date,
        confidence_thresholds=confidence_thresholds,
    )
    return predictor.predict_match(
        home_team=home_team,
        away_team=away_team,
        venue_mode=venue_mode,
        prediction_date=prediction_date,
    )


def _clean_team_name(value: str, field_name: str) -> str:
    if value is None:
        raise DataValidationError(f"{field_name} is required")
    cleaned = str(value).strip()
    if not cleaned:
        raise DataValidationError(f"{field_name} is required")
    return cleaned


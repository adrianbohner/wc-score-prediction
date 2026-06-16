from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wc_predictor.data import DataValidationError, get_completed_matches
from wc_predictor.features.elo import InternalEloCalculator
from wc_predictor.features.rolling_features import (
    ROLLING_DIFF_COLUMNS,
    RollingFeatureBuilder,
    build_team_match_table,
)


VENUE_MODES = {
    "neutral": "neutral",
    "Neutral": "neutral",
    "team_1_host_advantage": "home_host",
    "Team 1 host advantage": "home_host",
    "team_2_host_advantage": "away_host",
    "Team 2 host advantage": "away_host",
}


@dataclass
class MatchFeatureBuilder:
    results: pd.DataFrame
    initial_elo: float = 1500.0

    def __post_init__(self) -> None:
        self.completed_results = get_completed_matches(self.results).sort_values(
            ["date", "match_id"], kind="mergesort"
        )
        self.team_match_table = build_team_match_table(self.completed_results)
        self.rolling_builder = RollingFeatureBuilder(self.team_match_table)
        self.elo_calculator = InternalEloCalculator(initial_rating=self.initial_elo)
        self.elo_matches = self.elo_calculator.fit_transform(self.completed_results)
        self.latest_elo = self.elo_calculator.get_ratings()
        self.team_universe = set(
            pd.concat([self.results["home_team"], self.results["away_team"]], ignore_index=True)
            .dropna()
            .astype(str)
            .str.strip()
        )

    def build_training_features(self) -> pd.DataFrame:
        if self.completed_results.empty:
            return pd.DataFrame()

        base = self.completed_results.reset_index(drop=True).copy()

        # Elo features: vectorised join on match_id
        elo_cols = self.elo_matches.set_index("match_id")[
            ["home_elo_pre", "away_elo_pre", "elo_diff", "elo_abs_diff"]
        ]
        base = base.join(elo_cols, on="match_id")

        # Rolling features: vectorised join from precomputed_stats_
        stats = self.rolling_builder.precomputed_stats_
        if not stats.empty:
            stats_df = stats.reset_index()
            fill = self.rolling_builder._global_mean_stats()

            home_df = (
                base[["match_id", "home_team"]]
                .rename(columns={"home_team": "team"})
                .merge(stats_df, on=["match_id", "team"], how="left")
                .drop(columns=["match_id", "team"])
                .fillna(fill)
                .reset_index(drop=True)
            )
            home_df.columns = [f"home_{c}" for c in home_df.columns]

            away_df = (
                base[["match_id", "away_team"]]
                .rename(columns={"away_team": "team"})
                .merge(stats_df, on=["match_id", "team"], how="left")
                .drop(columns=["match_id", "team"])
                .fillna(fill)
                .reset_index(drop=True)
            )
            away_df.columns = [f"away_{c}" for c in away_df.columns]

            base = pd.concat([base, home_df, away_df], axis=1)

        # Diff features
        for name in ROLLING_DIFF_COLUMNS:
            base[f"{name}_diff"] = base[f"home_{name}"] - base[f"away_{name}"]

        # Venue features
        neutral = base["neutral"].astype(bool)
        base["is_neutral"] = neutral
        base["home_host_advantage"] = (base["country"] == base["home_team"]) & ~neutral
        base["away_host_advantage"] = (base["country"] == base["away_team"]) & ~neutral
        base = self._add_tournament_features(base)

        return base

    def build_match_features(
        self,
        home_team: str,
        away_team: str,
        venue_mode: str = "neutral",
        prediction_date: str | pd.Timestamp | None = None,
        tournament: str = "FIFA World Cup",
    ) -> pd.DataFrame:
        self._validate_team(home_team)
        self._validate_team(away_team)
        if home_team == away_team:
            raise DataValidationError("home_team and away_team must be different")

        date = self._resolve_prediction_date(prediction_date)
        ratings = self.elo_calculator.get_ratings_as_of(date)
        home_elo = ratings.get(home_team, self.initial_elo)
        away_elo = ratings.get(away_team, self.initial_elo)
        features: dict[str, object] = {
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "tournament": tournament,
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
            "elo_diff": home_elo - away_elo,
            "elo_abs_diff": abs(home_elo - away_elo),
        }
        features.update(self._rolling_match_features(home_team, away_team, date))
        features.update(self._venue_features_from_mode(venue_mode))
        return self._add_tournament_features(pd.DataFrame([features]))

    def _rolling_match_features(
        self,
        home_team: str,
        away_team: str,
        date: pd.Timestamp,
    ) -> dict[str, object]:
        home_features = self.rolling_builder.build_features_as_of(home_team, date)
        away_features = self.rolling_builder.build_features_as_of(away_team, date)

        features: dict[str, object] = {}
        for name, value in home_features.items():
            features[f"home_{name}"] = value
        for name, value in away_features.items():
            features[f"away_{name}"] = value

        for name in ROLLING_DIFF_COLUMNS:
            features[f"{name}_diff"] = float(home_features[name]) - float(away_features[name])
        return features

    def _resolve_prediction_date(
        self,
        prediction_date: str | pd.Timestamp | None,
    ) -> pd.Timestamp:
        if prediction_date is not None:
            return pd.Timestamp(prediction_date)
        if self.completed_results.empty:
            return pd.Timestamp.today().normalize()
        return pd.Timestamp(self.completed_results["date"].max()) + pd.Timedelta(days=1)

    def _validate_team(self, team: str) -> None:
        if team not in self.team_universe:
            raise DataValidationError(f"unknown team: {team}")

    @staticmethod
    def _venue_features_from_match(
        country: str,
        home_team: str,
        away_team: str,
        neutral: bool,
    ) -> dict[str, bool]:
        return {
            "is_neutral": neutral,
            "home_host_advantage": (country == home_team) and not neutral,
            "away_host_advantage": (country == away_team) and not neutral,
        }

    @staticmethod
    def _venue_features_from_mode(venue_mode: str) -> dict[str, bool]:
        normalized = VENUE_MODES.get(venue_mode)
        if normalized is None:
            raise DataValidationError(f"invalid venue mode: {venue_mode}")

        return {
            "is_neutral": normalized == "neutral",
            "home_host_advantage": normalized == "home_host",
            "away_host_advantage": normalized == "away_host",
        }

    @staticmethod
    def _add_tournament_features(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        tournament = result["tournament"].fillna("").astype(str).str.lower()
        result["is_world_cup"] = tournament.eq("fifa world cup")
        result["is_friendly"] = tournament.str.contains("friendly", regex=False)
        result["is_qualifier"] = tournament.str.contains("qualif", regex=False)
        return result

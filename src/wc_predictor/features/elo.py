from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from wc_predictor.data import get_completed_matches


@dataclass
class InternalEloCalculator:
    initial_rating: float = 1500.0
    base_k: float = 30.0
    min_k: float = 8.0
    max_k: float = 40.0
    home_advantage: float = 0.0
    ratings_: dict[str, float] = field(default_factory=dict)
    rating_snapshots_: list[tuple[pd.Timestamp, dict[str, float]]] = field(default_factory=list)

    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        completed = get_completed_matches(matches).sort_values(
            ["date", "match_id"], kind="mergesort"
        )
        self.ratings_ = {}
        self.rating_snapshots_ = []

        output_groups: list[pd.DataFrame] = []
        for _, date_group in completed.groupby("date", sort=True):
            group = date_group.copy()
            pre_rows = []
            for row in group.itertuples(index=False):
                home_rating = self.get_rating(str(row.home_team))
                away_rating = self.get_rating(str(row.away_team))
                pre_rows.append(
                    {
                        "match_id": row.match_id,
                        "home_elo_pre": home_rating,
                        "away_elo_pre": away_rating,
                        "elo_diff": home_rating - away_rating,
                        "elo_abs_diff": abs(home_rating - away_rating),
                    }
                )

            pre_df = pd.DataFrame(pre_rows)
            group = group.merge(pre_df, on="match_id", how="left")
            output_groups.append(group)

            for row in date_group.itertuples(index=False):
                self._update_match(row)
            self.rating_snapshots_.append(
                (pd.Timestamp(date_group["date"].iloc[0]), dict(self.ratings_))
            )

        if not output_groups:
            return completed.assign(
                home_elo_pre=pd.Series(dtype=float),
                away_elo_pre=pd.Series(dtype=float),
                elo_diff=pd.Series(dtype=float),
                elo_abs_diff=pd.Series(dtype=float),
            )
        return pd.concat(output_groups, ignore_index=True)

    def get_rating(self, team: str) -> float:
        return self.ratings_.get(team, self.initial_rating)

    def get_ratings(self) -> dict[str, float]:
        return dict(self.ratings_)

    def get_ratings_as_of(self, date: str | pd.Timestamp) -> dict[str, float]:
        """Return ratings after matches strictly before ``date`` have been applied."""
        as_of_date = pd.Timestamp(date)
        ratings: dict[str, float] = {}
        for snapshot_date, snapshot in self.rating_snapshots_:
            if snapshot_date >= as_of_date:
                break
            ratings = snapshot
        return dict(ratings)

    def _update_match(self, row: object) -> None:
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        expected_home = self._expected_home(home_rating, away_rating)
        expected_away = 1.0 - expected_home

        home_score = float(row.home_score)
        away_score = float(row.away_score)
        if home_score > away_score:
            actual_home, actual_away = 1.0, 0.0
        elif home_score < away_score:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        k_factor = self._k_factor()
        multiplier = self._goal_difference_multiplier(home_score, away_score)

        self.ratings_[home_team] = home_rating + k_factor * multiplier * (
            actual_home - expected_home
        )
        self.ratings_[away_team] = away_rating + k_factor * multiplier * (
            actual_away - expected_away
        )

    def _expected_home(self, home_rating: float, away_rating: float) -> float:
        adjusted_diff = away_rating - home_rating - self.home_advantage
        return 1.0 / (1.0 + 10.0 ** (adjusted_diff / 400.0))

    def _k_factor(self) -> float:
        return max(self.min_k, min(self.max_k, self.base_k))

    @staticmethod
    def _goal_difference_multiplier(home_score: float, away_score: float) -> float:
        goal_difference = abs(home_score - away_score)
        if goal_difference == 0:
            return 1.0
        return min(2.0, 1.0 + 0.25 * (goal_difference - 1.0))

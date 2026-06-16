from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wc_predictor.data import get_completed_matches


ROLLING_COLUMNS = [
    "points_avg_5",
    "points_avg_10",
    "goals_for_avg_5",
    "goals_for_avg_10",
    "goals_against_avg_5",
    "goals_against_avg_10",
    "goal_diff_avg_10",
    "attack_strength_10",
    "defense_strength_10",
    "win_rate_5",
    "win_rate_10",
    "unbeaten_rate_10",
    "clean_sheet_rate_10",
    "scored_rate_10",
    "days_since_match",
    "matches_last_30",
    "matches_last_365",
    "matches_available_10",
    "low_history_flag",
]

# Subset used to compute home-away difference features in the training table.
ROLLING_DIFF_COLUMNS = [
    "points_avg_5",
    "points_avg_10",
    "goals_for_avg_10",
    "goals_against_avg_10",
    "goal_diff_avg_10",
    "attack_strength_10",
    "defense_strength_10",
    "win_rate_10",
    "unbeaten_rate_10",
    "clean_sheet_rate_10",
    "scored_rate_10",
    "days_since_match",
    "matches_last_30",
    "matches_last_365",
]

REST_DAYS_DEFAULT = 365.0
REST_DAYS_CAP = 365.0


def build_team_match_table(results: pd.DataFrame) -> pd.DataFrame:
    completed = get_completed_matches(results).sort_values(
        ["date", "match_id"], kind="mergesort"
    )
    rows: list[dict[str, object]] = []
    for row in completed.itertuples(index=False):
        home_score = float(row.home_score)
        away_score = float(row.away_score)
        rows.append(
            _team_row(
                match=row,
                team=str(row.home_team),
                opponent=str(row.away_team),
                is_home=True,
                goals_for=home_score,
                goals_against=away_score,
            )
        )
        rows.append(
            _team_row(
                match=row,
                team=str(row.away_team),
                opponent=str(row.home_team),
                is_home=False,
                goals_for=away_score,
                goals_against=home_score,
            )
        )

    return pd.DataFrame(rows).sort_values(["date", "match_id", "team"], kind="mergesort")


@dataclass
class RollingFeatureBuilder:
    team_match_table: pd.DataFrame

    def __post_init__(self) -> None:
        self.team_match_table = self.team_match_table.sort_values(
            ["date", "match_id", "team"], kind="mergesort"
        ).reset_index(drop=True)
        self.team_histories = {
            team: group.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
            for team, group in self.team_match_table.groupby("team", sort=False)
        }
        self.global_goals_for_avg = _safe_mean(self.team_match_table, "goals_for", 1.25)
        self.global_goals_against_avg = _safe_mean(
            self.team_match_table, "goals_against", self.global_goals_for_avg
        )
        self.global_points_avg = _safe_mean(self.team_match_table, "points", 1.0)
        self.global_goal_diff_avg = _safe_mean(self.team_match_table, "goal_diff", 0.0)
        self.precomputed_stats_ = self._build_precomputed_stats()

    def _build_precomputed_stats(self) -> pd.DataFrame:
        """Rolling stats for every (match_id, team) in the training set.

        Each match only sees results from strictly prior match dates. The index is
        (match_id, team) for O(1) lookup in build_training_features.
        """
        dfs = []

        for team, hist in self.team_histories.items():
            if hist.empty:
                continue

            g = hist.sort_values(["date", "match_id"]).reset_index(drop=True)
            start = 0
            for date, date_group in g.groupby("date", sort=True):
                history = g.iloc[:start]
                stats = self._stats_from_history(history, pd.Timestamp(date))
                dfs.append(
                    pd.DataFrame(
                        {
                            "match_id": date_group["match_id"].to_numpy(),
                            "team": team,
                            **stats,
                        }
                    )
                )
                start += len(date_group)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True).set_index(["match_id", "team"])

    def _global_mean_stats(self) -> dict[str, float | int | bool]:
        return {
            "points_avg_5": self.global_points_avg,
            "points_avg_10": self.global_points_avg,
            "goals_for_avg_5": self.global_goals_for_avg,
            "goals_for_avg_10": self.global_goals_for_avg,
            "goals_against_avg_5": self.global_goals_against_avg,
            "goals_against_avg_10": self.global_goals_against_avg,
            "goal_diff_avg_10": self.global_goal_diff_avg,
            "attack_strength_10": 1.0,
            "defense_strength_10": 1.0,
            "win_rate_5": 0.0,
            "win_rate_10": 0.0,
            "unbeaten_rate_10": 0.0,
            "clean_sheet_rate_10": 0.0,
            "scored_rate_10": 0.0,
            "days_since_match": REST_DAYS_DEFAULT,
            "matches_last_30": 0,
            "matches_last_365": 0,
            "matches_available_10": 0,
            "low_history_flag": True,
        }

    def build_features_as_of(self, team: str, date: pd.Timestamp) -> dict[str, float | bool]:
        as_of_date = pd.Timestamp(date)
        team_history = self.team_histories.get(team)
        if team_history is None:
            history = self.team_match_table.iloc[0:0]
        else:
            history = team_history[team_history["date"] < as_of_date]

        return self._stats_from_history(history, as_of_date)

    def _stats_from_history(
        self,
        history: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> dict[str, float | int | bool]:
        last_5 = history.tail(5)
        last_10 = history.tail(10)

        matches_available_10 = int(len(last_10))
        goals_for_avg_10 = _window_mean(last_10, "goals_for", self.global_goals_for_avg)
        goals_against_avg_10 = _window_mean(
            last_10, "goals_against", self.global_goals_against_avg
        )

        return {
            "points_avg_5": _window_mean(last_5, "points", self.global_points_avg),
            "points_avg_10": _window_mean(last_10, "points", self.global_points_avg),
            "goals_for_avg_5": _window_mean(last_5, "goals_for", self.global_goals_for_avg),
            "goals_for_avg_10": goals_for_avg_10,
            "goals_against_avg_5": _window_mean(
                last_5, "goals_against", self.global_goals_against_avg
            ),
            "goals_against_avg_10": goals_against_avg_10,
            "goal_diff_avg_10": _window_mean(last_10, "goal_diff", self.global_goal_diff_avg),
            "attack_strength_10": goals_for_avg_10 / max(self.global_goals_for_avg, 0.01),
            "defense_strength_10": goals_against_avg_10
            / max(self.global_goals_against_avg, 0.01),
            "win_rate_5": _window_mean(last_5, "win", 0.0),
            "win_rate_10": _window_mean(last_10, "win", 0.0),
            "unbeaten_rate_10": _window_mean(last_10, "unbeaten", 0.0),
            "clean_sheet_rate_10": _window_mean(last_10, "clean_sheet", 0.0),
            "scored_rate_10": _window_mean(last_10, "scored", 0.0),
            "days_since_match": _days_since_last_match(history, as_of_date),
            "matches_last_30": _matches_since(history, as_of_date, days=30),
            "matches_last_365": _matches_since(history, as_of_date, days=365),
            "matches_available_10": matches_available_10,
            "low_history_flag": matches_available_10 < 10,
        }


def _team_row(
    match: object,
    team: str,
    opponent: str,
    is_home: bool,
    goals_for: float,
    goals_against: float,
) -> dict[str, object]:
    if goals_for > goals_against:
        points = 3
        result = "W"
    elif goals_for < goals_against:
        points = 0
        result = "L"
    else:
        points = 1
        result = "D"

    return {
        "match_id": match.match_id,
        "date": match.date,
        "team": team,
        "opponent": opponent,
        "is_home": is_home,
        "is_away": not is_home,
        "is_neutral": bool(match.neutral),
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
        "points": points,
        "win": goals_for > goals_against,
        "draw": goals_for == goals_against,
        "loss": goals_for < goals_against,
        "unbeaten": goals_for >= goals_against,
        "clean_sheet": goals_against == 0,
        "scored": goals_for > 0,
        "result": result,
        "tournament": match.tournament,
    }


def _window_mean(window: pd.DataFrame, column: str, default: float) -> float:
    if window.empty:
        return float(default)
    return float(window[column].mean())


def _safe_mean(df: pd.DataFrame, column: str, default: float) -> float:
    if df.empty:
        return float(default)
    value = df[column].mean()
    if pd.isna(value):
        return float(default)
    return float(value)


def _days_since_last_match(history: pd.DataFrame, as_of_date: pd.Timestamp) -> float:
    if history.empty:
        return REST_DAYS_DEFAULT
    days = (pd.Timestamp(as_of_date) - pd.Timestamp(history["date"].max())).days
    return float(min(max(days, 0), REST_DAYS_CAP))


def _matches_since(history: pd.DataFrame, as_of_date: pd.Timestamp, days: int) -> int:
    if history.empty:
        return 0
    lower_bound = pd.Timestamp(as_of_date) - pd.Timedelta(days=days)
    return int((history["date"] >= lower_bound).sum())

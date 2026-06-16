from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import make_test_dir
from wc_predictor.data import DataValidationError, load_results
from wc_predictor.features import (
    MatchFeatureBuilder,
    RollingFeatureBuilder,
    build_team_match_table,
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
2020-01-02,Alpha,Gamma,1,1,Friendly,B City,Nowhere,true
2020-01-02,Beta,Gamma,3,0,Friendly,C City,Nowhere,true
2020-01-03,Alpha,Beta,0,1,Friendly,D City,Alpha,false
""",
    )


def test_team_match_table_has_one_row_per_team_per_completed_match() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))

    table = build_team_match_table(results)

    assert len(table) == 8
    first_home = table[(table["match_id"] == "2020-01-01|Alpha|Beta|A City") & (table["team"] == "Alpha")].iloc[0]
    first_away = table[(table["match_id"] == "2020-01-01|Alpha|Beta|A City") & (table["team"] == "Beta")].iloc[0]
    assert first_home["goals_for"] == 2
    assert first_home["points"] == 3
    assert first_away["goals_against"] == 2
    assert first_away["points"] == 0


def test_rolling_features_exclude_current_and_same_date_matches() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    table = build_team_match_table(results)
    builder = RollingFeatureBuilder(table)

    alpha_features = builder.build_features_as_of("Alpha", pd.Timestamp("2020-01-02"))
    gamma_features = builder.build_features_as_of("Gamma", pd.Timestamp("2020-01-02"))

    assert alpha_features["matches_available_10"] == 1
    assert alpha_features["goals_for_avg_10"] == 2
    assert gamma_features["matches_available_10"] == 0
    assert gamma_features["low_history_flag"] is True


def test_match_feature_builder_builds_training_rows_with_required_columns() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))

    features = MatchFeatureBuilder(results).build_training_features()

    required_columns = {
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff",
        "elo_abs_diff",
        "home_points_avg_5",
        "away_points_avg_5",
        "home_goals_for_avg_10",
        "away_goals_against_avg_10",
        "home_attack_strength_10",
        "away_defense_strength_10",
        "home_win_rate_10",
        "away_clean_sheet_rate_10",
        "home_days_since_match",
        "matches_last_365_diff",
        "home_matches_available_10",
        "away_low_history_flag",
        "is_neutral",
        "home_host_advantage",
        "away_host_advantage",
        "is_world_cup",
    }

    assert required_columns.issubset(features.columns)
    assert len(features) == 4


def test_training_features_exclude_current_match_history() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))

    features = MatchFeatureBuilder(results).build_training_features()
    first_match = features.loc[features["match_id"] == "2020-01-01|Alpha|Beta|A City"].iloc[0]

    assert first_match["home_matches_available_10"] == 0
    assert first_match["away_matches_available_10"] == 0
    assert bool(first_match["home_low_history_flag"]) is True


def test_build_match_features_for_arbitrary_ui_matchup() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    builder = MatchFeatureBuilder(results)

    features = builder.build_match_features(
        "Alpha",
        "Beta",
        venue_mode="Team 1 host advantage",
        prediction_date="2026-06-01",
    )

    row = features.iloc[0]
    assert row["home_team"] == "Alpha"
    assert row["away_team"] == "Beta"
    assert row["home_elo_pre"] != 1500
    assert row["home_days_since_match"] > 0
    assert bool(row["is_world_cup"]) is True
    assert bool(row["is_neutral"]) is False
    assert bool(row["home_host_advantage"]) is True
    assert bool(row["away_host_advantage"]) is False


def test_build_match_features_uses_elo_as_of_prediction_date() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    builder = MatchFeatureBuilder(results)

    same_day = builder.build_match_features(
        "Alpha",
        "Beta",
        prediction_date="2020-01-01",
    )
    after_first_match = builder.build_match_features(
        "Alpha",
        "Beta",
        prediction_date="2020-01-02",
    )

    assert same_day.loc[0, "home_elo_pre"] == 1500
    assert same_day.loc[0, "away_elo_pre"] == 1500
    assert after_first_match.loc[0, "home_elo_pre"] > 1500
    assert after_first_match.loc[0, "away_elo_pre"] < 1500


def test_build_match_features_rejects_unknown_team() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    builder = MatchFeatureBuilder(results)

    with pytest.raises(DataValidationError, match="unknown team: Atlantis"):
        builder.build_match_features("Alpha", "Atlantis")


def test_build_match_features_allows_future_only_team_with_default_history() -> None:
    tmp_path = make_test_dir()
    results = load_results(
        write_file(
            tmp_path,
            "results.csv",
            """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2020-01-01,Alpha,Beta,2,0,Friendly,A City,Nowhere,true
2026-06-11,Futureland,Alpha,,,FIFA World Cup,B City,Nowhere,true
""",
        )
    )
    builder = MatchFeatureBuilder(results)

    features = builder.build_match_features(
        "Futureland",
        "Alpha",
        venue_mode="neutral",
        prediction_date="2026-06-01",
    )

    assert features.loc[0, "home_team"] == "Futureland"
    assert features.loc[0, "home_matches_available_10"] == 0
    assert bool(features.loc[0, "home_low_history_flag"]) is True


def test_build_match_features_rejects_invalid_venue_mode() -> None:
    tmp_path = make_test_dir()
    results = load_results(sample_results(tmp_path))
    builder = MatchFeatureBuilder(results)

    with pytest.raises(DataValidationError, match="invalid venue mode"):
        builder.build_match_features("Alpha", "Beta", venue_mode="Mars")

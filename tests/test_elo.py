from pathlib import Path

import pytest

from tests.helpers import make_test_dir
from wc_predictor.data import load_results
from wc_predictor.features import InternalEloCalculator


def write_file(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_elo_before_first_match_equals_initial_rating() -> None:
    tmp_path = make_test_dir()
    results = load_results(
        write_file(
            tmp_path,
            "results.csv",
            """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2020-01-01,Alpha,Beta,2,0,Friendly,A City,Nowhere,true
""",
        )
    )

    rated = InternalEloCalculator(initial_rating=1500).fit_transform(results)

    assert rated.loc[0, "home_elo_pre"] == 1500
    assert rated.loc[0, "away_elo_pre"] == 1500
    assert rated.loc[0, "elo_diff"] == 0


def test_elo_winner_gains_and_loser_loses_points() -> None:
    tmp_path = make_test_dir()
    results = load_results(
        write_file(
            tmp_path,
            "results.csv",
            """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2020-01-01,Alpha,Beta,2,0,Friendly,A City,Nowhere,true
""",
        )
    )
    calculator = InternalEloCalculator(initial_rating=1500)

    calculator.fit_transform(results)
    ratings = calculator.get_ratings()

    assert ratings["Alpha"] > 1500
    assert ratings["Beta"] < 1500
    assert ratings["Alpha"] - 1500 == pytest.approx(1500 - ratings["Beta"])


def test_elo_same_date_matches_do_not_update_each_other_pre_ratings() -> None:
    tmp_path = make_test_dir()
    results = load_results(
        write_file(
            tmp_path,
            "results.csv",
            """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2020-01-01,Alpha,Beta,2,0,Friendly,A City,Nowhere,true
2020-01-01,Alpha,Gamma,0,1,Friendly,B City,Nowhere,true
""",
        )
    )

    rated = InternalEloCalculator(initial_rating=1500).fit_transform(results)

    assert rated.loc[0, "home_elo_pre"] == 1500
    assert rated.loc[1, "home_elo_pre"] == 1500
    assert rated.loc[1, "away_elo_pre"] == 1500


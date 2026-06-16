from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import make_test_dir
from wc_predictor.data import (
    DataValidationError,
    find_duplicate_match_ids,
    get_completed_matches,
    load_goalscorers,
    load_results,
)


def write_csv(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def valid_results_csv() -> str:
    return """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,Al Khor,Qatar,false
2022-11-21,England,Iran,6,2,FIFA World Cup,Doha,Qatar,true
2026-06-11,Mexico,Canada,,,FIFA World Cup,Mexico City,Mexico,false
"""


def test_load_results_parses_dates_scores_booleans_and_match_ids() -> None:
    tmp_path = make_test_dir()
    path = write_csv(tmp_path, "results.csv", valid_results_csv())

    df = load_results(path)

    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.loc[0, "home_score"] == 0
    assert bool(df.loc[0, "neutral"]) is False
    assert bool(df.loc[1, "neutral"]) is True
    assert df.loc[0, "match_id"] == "2022-11-20|Qatar|Ecuador|Al Khor"


def test_get_completed_matches_filters_future_rows() -> None:
    tmp_path = make_test_dir()
    path = write_csv(tmp_path, "results.csv", valid_results_csv())
    df = load_results(path)

    completed = get_completed_matches(df)

    assert len(completed) == 2
    assert completed["home_score"].notna().all()
    assert completed["away_score"].notna().all()


def test_load_results_rejects_missing_columns() -> None:
    tmp_path = make_test_dir()
    path = write_csv(
        tmp_path,
        "results.csv",
        """
date,home_team,away_team,home_score,away_score
2022-11-20,Qatar,Ecuador,0,2
""",
    )

    with pytest.raises(DataValidationError, match="missing required columns"):
        load_results(path)


def test_load_results_rejects_invalid_dates() -> None:
    tmp_path = make_test_dir()
    path = write_csv(
        tmp_path,
        "results.csv",
        """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
not-a-date,Qatar,Ecuador,0,2,FIFA World Cup,Al Khor,Qatar,false
""",
    )

    with pytest.raises(DataValidationError, match="invalid dates"):
        load_results(path)


def test_load_results_rejects_negative_scores() -> None:
    tmp_path = make_test_dir()
    path = write_csv(
        tmp_path,
        "results.csv",
        """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2022-11-20,Qatar,Ecuador,-1,2,FIFA World Cup,Al Khor,Qatar,false
""",
    )

    with pytest.raises(DataValidationError, match="negative scores"):
        load_results(path)


def test_load_results_rejects_partial_score_rows() -> None:
    tmp_path = make_test_dir()
    path = write_csv(
        tmp_path,
        "results.csv",
        """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2026-06-11,Mexico,Canada,1,,FIFA World Cup,Mexico City,Mexico,false
""",
    )

    with pytest.raises(DataValidationError, match="both scores present or both scores missing"):
        load_results(path)


def test_duplicate_match_ids_are_reported() -> None:
    tmp_path = make_test_dir()
    path = write_csv(
        tmp_path,
        "results.csv",
        """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,Al Khor,Qatar,false
2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,Al Khor,Qatar,false
""",
    )
    df = load_results(path)

    duplicates = find_duplicate_match_ids(df)

    assert len(duplicates) == 2
    assert duplicates["match_id"].nunique() == 1


def test_load_goalscorers_parses_expected_types() -> None:
    tmp_path = make_test_dir()
    path = write_csv(
        tmp_path,
        "goalscorers.csv",
        """
date,home_team,away_team,team,scorer,minute,own_goal,penalty
2022-11-20,Qatar,Ecuador,Ecuador,Enner Valencia,16,false,true
""",
    )

    df = load_goalscorers(path)

    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.loc[0, "minute"] == 16
    assert bool(df.loc[0, "own_goal"]) is False
    assert bool(df.loc[0, "penalty"]) is True


def test_load_goalscorers_rejects_missing_columns() -> None:
    tmp_path = make_test_dir()
    path = write_csv(
        tmp_path,
        "goalscorers.csv",
        """
date,home_team,away_team,team,scorer
2022-11-20,Qatar,Ecuador,Ecuador,Enner Valencia
""",
    )

    with pytest.raises(DataValidationError, match="missing required columns"):
        load_goalscorers(path)

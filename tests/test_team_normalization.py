from pathlib import Path

import pytest

from tests.helpers import make_test_dir
from wc_predictor.data import (
    DataValidationError,
    TeamNameNormalizer,
    get_team_universe,
    load_goalscorers,
    load_results,
    load_selectable_teams,
    load_team_name_map,
    validate_goalscorer_teams,
    validate_selectable_teams,
)


def write_file(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def write_results(tmp_path: Path) -> Path:
    return write_file(
        tmp_path,
        "results.csv",
        """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2022-11-20,USA,Canada,1,0,Friendly,Columbus,United States,false
2022-11-21,Czechia,Germany,2,2,Friendly,Prague,Czechia,false
2022-11-22,France,Spain,1,1,Friendly,Paris,France,false
""",
    )


def test_load_team_name_map_reads_yaml_mapping() -> None:
    tmp_path = make_test_dir()
    path = write_file(
        tmp_path,
        "team_name_map.yaml",
        """
USA: United States
Czechia: Czech Republic
""",
    )

    mapping = load_team_name_map(path)

    assert mapping == {"USA": "United States", "Czechia": "Czech Republic"}


def test_normalize_results_applies_aliases_and_rebuilds_match_id() -> None:
    tmp_path = make_test_dir()
    results = load_results(write_results(tmp_path))
    normalizer = TeamNameNormalizer({"USA": "United States", "Czechia": "Czech Republic"})

    normalized = normalizer.normalize_results(results)

    assert normalized.loc[0, "home_team"] == "United States"
    assert normalized.loc[1, "home_team"] == "Czech Republic"
    assert normalized.loc[0, "match_id"] == "2022-11-20|United States|Canada|Columbus"


def test_normalize_goalscorers_applies_same_aliases() -> None:
    tmp_path = make_test_dir()
    path = write_file(
        tmp_path,
        "goalscorers.csv",
        """
date,home_team,away_team,team,scorer,minute,own_goal,penalty
2022-11-20,USA,Canada,USA,Player One,10,false,false
""",
    )
    goalscorers = load_goalscorers(path)
    normalizer = TeamNameNormalizer({"USA": "United States"})

    normalized = normalizer.normalize_goalscorers(goalscorers)

    assert normalized.loc[0, "home_team"] == "United States"
    assert normalized.loc[0, "team"] == "United States"


def test_load_selectable_teams_normalizes_and_sorts() -> None:
    tmp_path = make_test_dir()
    path = write_file(
        tmp_path,
        "world_cup_2026_teams.yaml",
        """
teams:
  - Spain
  - USA
  - Canada
  - United States
""",
    )
    normalizer = TeamNameNormalizer({"USA": "United States"})

    teams = load_selectable_teams(path, normalizer)

    assert teams == ["Canada", "Spain", "United States"]


def test_validate_selectable_teams_accepts_known_teams() -> None:
    tmp_path = make_test_dir()
    results = load_results(write_results(tmp_path))
    normalizer = TeamNameNormalizer({"USA": "United States", "Czechia": "Czech Republic"})
    normalized_results = normalizer.normalize_results(results)

    validate_selectable_teams(
        ["Canada", "Czech Republic", "United States"],
        get_team_universe(normalized_results),
    )


def test_validate_selectable_teams_rejects_unknown_team() -> None:
    tmp_path = make_test_dir()
    results = load_results(write_results(tmp_path))
    normalizer = TeamNameNormalizer({"USA": "United States"})
    normalized_results = normalizer.normalize_results(results)

    with pytest.raises(DataValidationError, match="missing from historical results: Atlantis"):
        validate_selectable_teams(["Atlantis"], get_team_universe(normalized_results))


def test_validate_goalscorer_teams_rejects_unknown_scorer_team() -> None:
    tmp_path = make_test_dir()
    results = load_results(write_results(tmp_path))
    goalscorers_path = write_file(
        tmp_path,
        "goalscorers.csv",
        """
date,home_team,away_team,team,scorer,minute,own_goal,penalty
2022-11-20,USA,Canada,Atlantis,Player One,10,false,false
""",
    )
    goalscorers = load_goalscorers(goalscorers_path)

    with pytest.raises(DataValidationError, match="missing from results.csv: Atlantis"):
        validate_goalscorer_teams(goalscorers, get_team_universe(results))

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
import yaml

from wc_predictor.data.validate_data import DataValidationError, add_match_id


class TeamNameNormalizer:
    def __init__(self, mapping: Mapping[str, str] | None = None):
        self.mapping = {
            self._clean_name(source): self._clean_name(target)
            for source, target in (mapping or {}).items()
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TeamNameNormalizer":
        return cls(load_team_name_map(path))

    def normalize_name(self, name: object) -> str:
        cleaned = self._clean_name(name)
        if not cleaned:
            raise DataValidationError("team name cannot be empty")
        return self.mapping.get(cleaned, cleaned)

    def normalize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column in ["home_team", "away_team"]:
            result[column] = result[column].map(self.normalize_name)

        if "match_id" in result.columns:
            result = result.drop(columns=["match_id"])
        if {"date", "home_team", "away_team", "city"}.issubset(result.columns):
            result = add_match_id(result)
        return result

    def normalize_goalscorers(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column in ["home_team", "away_team", "team"]:
            result[column] = result[column].map(self.normalize_name)
        return result

    @staticmethod
    def _clean_name(name: object) -> str:
        if pd.isna(name):
            return ""
        return str(name).strip()


def load_team_name_map(path: str | Path) -> dict[str, str]:
    config_path = Path(path)
    if not config_path.exists():
        raise DataValidationError(f"team name map was not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise DataValidationError("team name map must be a YAML mapping")

    return {str(source): str(target) for source, target in raw_config.items()}


def load_selectable_teams(
    path: str | Path,
    normalizer: TeamNameNormalizer | None = None,
) -> list[str]:
    config_path = Path(path)
    if not config_path.exists():
        raise DataValidationError(f"selectable team config was not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    teams = raw_config.get("teams")
    if not isinstance(teams, list):
        raise DataValidationError("selectable team config must contain a teams list")

    normalizer = normalizer or TeamNameNormalizer()
    normalized = [normalizer.normalize_name(team) for team in teams]
    return sorted(set(normalized))


def get_team_universe(results: pd.DataFrame) -> set[str]:
    teams = pd.concat([results["home_team"], results["away_team"]], ignore_index=True)
    return {str(team).strip() for team in teams.dropna() if str(team).strip()}


def validate_selectable_teams(
    selectable_teams: Iterable[str],
    team_universe: Iterable[str],
) -> None:
    universe = set(team_universe)
    missing = sorted(set(selectable_teams) - universe)
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"selectable teams are missing from historical results: {missing_text}"
        )


def validate_goalscorer_teams(
    goalscorers: pd.DataFrame,
    team_universe: Iterable[str],
) -> None:
    universe = set(team_universe)
    scorer_teams = set(goalscorers["team"].dropna().astype(str).str.strip())
    missing = sorted(scorer_teams - universe)
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"goalscorers.csv contains teams missing from results.csv: {missing_text}"
        )


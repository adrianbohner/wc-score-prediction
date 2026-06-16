"""Data loading, validation, and normalization helpers."""

from wc_predictor.data.load_data import load_goalscorers, load_results
from wc_predictor.data.normalize_teams import (
    TeamNameNormalizer,
    get_team_universe,
    load_selectable_teams,
    load_team_name_map,
    validate_goalscorer_teams,
    validate_selectable_teams,
)
from wc_predictor.data.validate_data import (
    DataValidationError,
    add_match_id,
    find_duplicate_match_ids,
    get_completed_matches,
    validate_goalscorers,
    validate_results,
)

__all__ = [
    "DataValidationError",
    "TeamNameNormalizer",
    "add_match_id",
    "find_duplicate_match_ids",
    "get_completed_matches",
    "get_team_universe",
    "load_goalscorers",
    "load_selectable_teams",
    "load_results",
    "load_team_name_map",
    "validate_goalscorer_teams",
    "validate_goalscorers",
    "validate_results",
    "validate_selectable_teams",
]

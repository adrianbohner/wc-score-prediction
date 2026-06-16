from pathlib import Path

import yaml

import wc_predictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_has_version() -> None:
    assert wc_predictor.__version__ == "0.1.0"


def test_starter_configs_exist() -> None:
    expected_paths = [
        PROJECT_ROOT / "configs" / "model_config.yaml",
        PROJECT_ROOT / "configs" / "team_name_map.yaml",
        PROJECT_ROOT / "configs" / "world_cup_2026_teams.yaml",
    ]

    for path in expected_paths:
        assert path.exists()


def test_world_cup_team_config_has_teams() -> None:
    path = PROJECT_ROOT / "configs" / "world_cup_2026_teams.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(config["teams"], list)
    assert "United States" in config["teams"]


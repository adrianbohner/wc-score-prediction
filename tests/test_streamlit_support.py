from pathlib import Path

import pytest

from tests.helpers import make_test_dir
from wc_predictor.data import DataValidationError
from wc_predictor.models.artifact import load_model_artifact
from wc_predictor.models.train_app_model import train_model_artifact
from wc_predictor.streamlit_support import (
    build_training_resources,
    build_streamlit_resources,
    filter_results_for_training,
    format_percent,
    load_confidence_thresholds,
    load_runtime_selectable_teams,
    load_yaml_config,
)


def write_file(tmp_path: Path, relative_path: str, content: str) -> Path:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def write_sample_project(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "configs/model_config.yaml",
        """
training:
  start_date: "2020-01-01"
  prediction_cutoff_date: "2026-06-01"
model:
  max_goals: 8
  alpha: 0.5
ui:
  confidence_thresholds:
    low_max: 0.12
    medium_max: 0.18
""",
    )
    write_file(
        tmp_path,
        "configs/team_name_map.yaml",
        """
USA: United States
""",
    )
    write_file(
        tmp_path,
        "configs/world_cup_2026_teams.yaml",
        """
teams:
  - Alpha
  - Beta
  - Gamma
""",
    )
    write_file(
        tmp_path,
        "data/raw/results.csv",
        """
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2019-01-01,Alpha,Beta,9,0,Friendly,Old City,Nowhere,true
2020-01-01,Alpha,Beta,2,0,Friendly,A City,Nowhere,true
2020-01-02,Gamma,Beta,1,1,Friendly,B City,Nowhere,true
2020-01-03,Alpha,Gamma,3,1,Friendly,C City,Nowhere,true
2020-01-04,Beta,Gamma,0,1,Friendly,D City,Nowhere,true
2020-01-05,Alpha,Beta,1,1,Friendly,E City,Alpha,false
2026-06-02,Alpha,Beta,9,0,Friendly,Future City,Nowhere,true
""",
    )


def test_load_yaml_config_requires_mapping() -> None:
    tmp_path = make_test_dir()
    path = write_file(tmp_path, "config.yaml", "- Alpha\n- Beta")

    with pytest.raises(DataValidationError, match="YAML mapping"):
        load_yaml_config(path)


def test_load_confidence_thresholds_reads_ui_config() -> None:
    thresholds = load_confidence_thresholds(
        {"ui": {"confidence_thresholds": {"low_max": 0.1, "medium_max": 0.2}}}
    )

    assert thresholds.low_max == 0.1
    assert thresholds.medium_max == 0.2


def test_format_percent() -> None:
    assert format_percent(0.1234) == "12.3%"


def test_build_training_resources_trains_predictor_and_filters_start_date() -> None:
    tmp_path = make_test_dir()
    write_sample_project(tmp_path)

    resources = build_training_resources(tmp_path)
    prediction = resources.predictor.predict_match(
        "Alpha",
        "Gamma",
        venue_mode="neutral",
        prediction_date=resources.feature_cutoff_date,
    )

    assert resources.teams == ["Alpha", "Beta", "Gamma"]
    assert resources.training_match_count == 5
    assert resources.feature_cutoff_date == "2026-06-01"
    assert resources.predictor.feature_builder.completed_results["date"].max().strftime("%Y-%m-%d") == "2020-01-05"
    assert prediction["most_likely_score"].startswith("Alpha ")
    assert prediction["top_scorelines"]


def test_filter_results_for_training_without_start_date_returns_original() -> None:
    tmp_path = make_test_dir()
    write_sample_project(tmp_path)
    resources = build_training_resources(tmp_path)

    unfiltered = filter_results_for_training(
        resources.predictor.feature_builder.completed_results,
        {"training": {}},
    )

    assert len(unfiltered) == resources.training_match_count


def test_train_model_artifact_writes_loadable_artifact() -> None:
    tmp_path = make_test_dir()
    write_sample_project(tmp_path)
    artifact_path = tmp_path / "models" / "match_score_model.pkl"

    written_path = train_model_artifact(tmp_path, artifact_path=artifact_path)
    resources = load_model_artifact(written_path)

    prediction = resources.predictor.predict_match(
        "Alpha",
        "Gamma",
        venue_mode="neutral",
        prediction_date=resources.feature_cutoff_date,
    )

    assert written_path == artifact_path
    assert resources.teams == ["Alpha", "Beta", "Gamma"]
    assert prediction["top_scorelines"]


def test_train_model_artifact_verbose_prints_progress(capsys: pytest.CaptureFixture[str]) -> None:
    tmp_path = make_test_dir()
    write_sample_project(tmp_path)
    artifact_path = tmp_path / "models" / "match_score_model.pkl"

    train_model_artifact(tmp_path, artifact_path=artifact_path, verbose=True)
    output = capsys.readouterr().out

    assert "World Cup score model training" in output
    assert "Loading historical results" in output
    assert "Building training feature table" in output
    assert "Model training complete" in output
    assert "Training complete" in output


def test_build_streamlit_resources_loads_existing_artifact() -> None:
    tmp_path = make_test_dir()
    write_sample_project(tmp_path)
    train_model_artifact(tmp_path, artifact_path=tmp_path / "models" / "match_score_model.pkl")

    resources = build_streamlit_resources(tmp_path)

    assert resources.training_match_count == 5
    assert resources.predictor.predict_match("Alpha", "Gamma")["most_likely_score"]


def test_build_streamlit_resources_uses_current_team_config_not_artifact_list() -> None:
    tmp_path = make_test_dir()
    write_sample_project(tmp_path)
    write_file(
        tmp_path,
        "configs/world_cup_2026_teams.yaml",
        """
teams:
  - Alpha
  - Beta
""",
    )
    train_model_artifact(tmp_path, artifact_path=tmp_path / "models" / "match_score_model.pkl")

    write_file(
        tmp_path,
        "configs/world_cup_2026_teams.yaml",
        """
teams:
  - Alpha
  - Beta
  - Gamma
""",
    )
    resources = build_streamlit_resources(tmp_path)

    assert resources.teams == ["Alpha", "Beta", "Gamma"]


def test_runtime_team_config_reports_stale_artifact_for_unsupported_team() -> None:
    tmp_path = make_test_dir()
    write_sample_project(tmp_path)
    artifact_path = tmp_path / "models" / "match_score_model.pkl"
    train_model_artifact(tmp_path, artifact_path=artifact_path)
    resources = load_model_artifact(artifact_path)
    write_file(
        tmp_path,
        "configs/world_cup_2026_teams.yaml",
        """
teams:
  - Alpha
  - Atlantis
""",
    )

    with pytest.raises(DataValidationError, match="model artifact is stale or incompatible"):
        load_runtime_selectable_teams(tmp_path, resources)


def test_build_streamlit_resources_requires_pretrained_artifact() -> None:
    tmp_path = make_test_dir()
    write_sample_project(tmp_path)

    with pytest.raises(DataValidationError, match="model artifact was not found"):
        build_streamlit_resources(tmp_path)

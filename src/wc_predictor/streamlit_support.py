from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from wc_predictor.data import (
    DataValidationError,
    TeamNameNormalizer,
    get_team_universe,
    load_results,
    load_selectable_teams,
    validate_selectable_teams,
)
from wc_predictor.features import MatchFeatureBuilder
from wc_predictor.features.weights import compute_sample_weights
from wc_predictor.models import MatchPredictor, train_score_model
from wc_predictor.models.train import tune_model_hyperparameters
from wc_predictor.models.artifact import ARTIFACT_VERSION, DEFAULT_MODEL_ARTIFACT_PATH, load_model_artifact
from wc_predictor.presentation import ConfidenceThresholds


@dataclass
class ModelArtifactMetadata:
    """Descriptive record stored inside every saved artifact."""

    model_type: str
    tuned_alpha: float
    train_cutoff: str
    n_matches: int
    created_at: str        # ISO-8601 UTC
    artifact_version: str
    tuned_reg_strength: float | None = None
    model_weights: tuple[float, ...] | None = None
    holdout_rps: float | None = None
    holdout_skill_score: float | None = None
    holdout_n_matches: int | None = None
    holdout_days: int | None = None

    def summary_lines(self) -> list[str]:
        lines = [
            f"Model type      : {self.model_type}",
            f"Tuned alpha     : {self.tuned_alpha}",
            f"Training matches: {self.n_matches:,}",
            f"Feature cutoff  : {self.train_cutoff}",
            f"Created at      : {self.created_at}",
            f"Artifact version: {self.artifact_version}",
        ]
        if self.tuned_reg_strength is not None:
            lines.append(f"Tuned reg       : {self.tuned_reg_strength}")
        if self.model_weights is not None:
            weights = ", ".join(f"{weight:.2f}" for weight in self.model_weights)
            lines.append(f"Model weights   : {weights}")
        if self.holdout_rps is not None:
            lines.append(
                f"Recent {self.holdout_days}d RPS : {self.holdout_rps:.4f} "
                f"(skill {self.holdout_skill_score:+.1%}, n={self.holdout_n_matches:,})"
            )
        return lines


@dataclass
class StreamlitAppResources:
    teams: list[str]
    predictor: MatchPredictor
    training_match_count: int
    feature_cutoff_date: str
    metadata: ModelArtifactMetadata | None = field(default=None)


def build_streamlit_resources(project_root: Path) -> StreamlitAppResources:
    artifact = load_model_artifact(project_root / DEFAULT_MODEL_ARTIFACT_PATH)
    teams = load_runtime_selectable_teams(project_root, artifact)
    return StreamlitAppResources(
        teams=teams,
        predictor=artifact.predictor,
        training_match_count=artifact.training_match_count,
        feature_cutoff_date=artifact.feature_cutoff_date,
        metadata=getattr(artifact, "metadata", None),
    )


ProgressCallback = Callable[[str], None]


def build_training_resources(
    project_root: Path,
    progress: ProgressCallback | None = None,
) -> StreamlitAppResources:
    _report(progress, "Loading model configuration")
    config = load_yaml_config(project_root / "configs" / "model_config.yaml")
    _report(progress, "Loading team-name mapping")
    normalizer = TeamNameNormalizer.from_yaml(project_root / "configs" / "team_name_map.yaml")

    _report(progress, "Loading historical results")
    results = load_results(project_root / "data" / "raw" / "results.csv")
    _report(progress, f"Loaded {len(results):,} result rows")
    _report(progress, "Normalizing team names")
    results = normalizer.normalize_results(results)
    unfiltered_count = len(results)
    results = filter_results_for_training(results, config)
    _report(
        progress,
        f"Using {len(results):,} result rows after training date filters "
        f"({unfiltered_count:,} before filter)",
    )

    _report(progress, "Loading selectable World Cup team list")
    teams = load_selectable_teams(
        project_root / "configs" / "world_cup_2026_teams.yaml",
        normalizer=normalizer,
    )
    _report(progress, f"Loaded {len(teams):,} selectable teams")
    _report(progress, "Validating selectable teams against historical results")
    validate_selectable_teams(teams, get_team_universe(results))

    _report(progress, "Building feature pipeline")
    feature_builder = MatchFeatureBuilder(results)
    _report(progress, "Building training feature table")
    training_features = feature_builder.build_training_features()
    if training_features.empty:
        raise DataValidationError("no completed matches are available for model training")
    _report(progress, f"Built {len(training_features):,} training feature rows")

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    rho_bounds = model_config.get("rho_bounds", [-0.3, 0.3])
    model_type = str(model_config.get("selected_type", "poisson"))
    max_goals = int(model_config.get("max_goals", 8))

    sample_weight = None
    if training_config.get("use_time_decay", True):
        cutoff_date = pd.Timestamp(
            training_config.get("prediction_cutoff_date")
            or training_features["date"].max()
        )
        sample_weight = compute_sample_weights(
            training_features,
            reference_date=cutoff_date,
            half_life_days=float(training_config.get("half_life_days", 1460.0)),
            friendly_weight=float(training_config.get("friendly_weight", 0.5)),
        )
        _report(progress, "Sample weights computed (time decay + tournament type)")

    alpha = float(model_config.get("alpha", 1.0))
    reg_strength = (
        float(model_config["reg_strength"]) if "reg_strength" in model_config else None
    )
    model_weights = _optional_float_tuple(model_config.get("model_weights"))
    if model_config.get("tune_alpha", True):
        _report(progress, "Tuning model hyperparameters via time-series CV")
        alpha_candidates = tuple(
            model_config.get("alpha_candidates", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
        )
        tuned = tune_model_hyperparameters(
            training_features=training_features,
            sample_weight=sample_weight,
            candidates=alpha_candidates,
            max_goals=max_goals,
            model_type=model_type,
            ensemble_weight_candidates=_ensemble_weight_candidates(model_config),
        )
        alpha = tuned.alpha
        reg_strength = tuned.reg_strength if tuned.reg_strength is not None else reg_strength
        model_weights = tuned.model_weights if tuned.model_weights is not None else model_weights
        _report(progress, f"Best alpha: {alpha}")
        if reg_strength is not None:
            _report(progress, f"Best regularisation: {reg_strength}")
        if model_weights is not None:
            _report(progress, f"Best ensemble weights: {list(model_weights)}")

    _report(
        progress,
        f"Training {model_type} model (max_goals={max_goals}, alpha={alpha})",
    )
    model = train_score_model(
        training_features=training_features,
        model_type=model_type,
        max_goals=max_goals,
        alpha=alpha,
        rho_bounds=(float(rho_bounds[0]), float(rho_bounds[1])),
        sample_weight=sample_weight,
        reg_strength=reg_strength,
        model_weights=model_weights,
    )
    _report(progress, "Model training complete")

    cutoff_date = str(
        config.get("training", {}).get("prediction_cutoff_date")
        or training_features["date"].max().strftime("%Y-%m-%d")
    )
    predictor = MatchPredictor(
        model=model,
        feature_builder=feature_builder,
        model_version=f"{model_type}-v1",
        feature_cutoff_date=cutoff_date,
        confidence_thresholds=load_confidence_thresholds(config),
    )

    metadata = ModelArtifactMetadata(
        model_type=model_type,
        tuned_alpha=alpha,
        train_cutoff=cutoff_date,
        n_matches=len(training_features),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        artifact_version=ARTIFACT_VERSION,
        tuned_reg_strength=reg_strength,
        model_weights=model_weights,
    )

    return StreamlitAppResources(
        teams=teams,
        predictor=predictor,
        training_match_count=len(training_features),
        feature_cutoff_date=cutoff_date,
        metadata=metadata,
    )


def load_runtime_selectable_teams(
    project_root: Path,
    resources: StreamlitAppResources,
) -> list[str]:
    normalizer = TeamNameNormalizer.from_yaml(project_root / "configs" / "team_name_map.yaml")
    teams = load_selectable_teams(
        project_root / "configs" / "world_cup_2026_teams.yaml",
        normalizer=normalizer,
    )
    try:
        validate_selectable_teams(teams, resources.predictor.feature_builder.team_universe)
    except DataValidationError as exc:
        raise DataValidationError(
            f"{exc}. The model artifact is stale or incompatible; retrain with "
            "`python -m wc_predictor.models.train_app_model`."
        ) from exc
    return teams


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DataValidationError(f"config file was not found at {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise DataValidationError(f"config file must contain a YAML mapping: {path}")
    return config


def filter_results_for_training(results: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    training_config = config.get("training", {})
    start_date = training_config.get("start_date")
    cutoff_date = training_config.get("prediction_cutoff_date")

    mask = pd.Series(True, index=results.index)
    if start_date:
        mask &= results["date"] >= pd.Timestamp(start_date)
    if cutoff_date:
        mask &= results["date"] <= pd.Timestamp(cutoff_date)
    return results.loc[mask].copy()


def load_confidence_thresholds(config: dict[str, Any]) -> ConfidenceThresholds:
    thresholds = config.get("ui", {}).get("confidence_thresholds", {})
    return ConfidenceThresholds(
        low_max=float(thresholds.get("low_max", 0.12)),
        medium_max=float(thresholds.get("medium_max", 0.18)),
    )


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _optional_float_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise DataValidationError("model_weights must be a list of numbers")
    return tuple(float(item) for item in value)


def _ensemble_weight_candidates(config: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    raw = config.get("ensemble_weight_candidates")
    if raw is None:
        return ((0.25, 0.75), (0.5, 0.5), (0.75, 0.25))
    if not isinstance(raw, list):
        raise DataValidationError("ensemble_weight_candidates must be a list of pairs")

    candidates = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise DataValidationError("ensemble_weight_candidates must contain weight pairs")
        candidates.append((float(item[0]), float(item[1])))
    return tuple(candidates)

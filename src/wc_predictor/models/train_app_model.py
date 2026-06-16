from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import numpy as np
import pandas as pd

from wc_predictor.data import TeamNameNormalizer, get_completed_matches, load_results
from wc_predictor.features import MatchFeatureBuilder
from wc_predictor.features.weights import compute_sample_weights
from wc_predictor.models import train_score_model
from wc_predictor.models.artifact import DEFAULT_MODEL_ARTIFACT_PATH, save_model_artifact
from wc_predictor.models.train import tune_model_hyperparameters
from wc_predictor.streamlit_support import (
    StreamlitAppResources,
    build_training_resources,
    filter_results_for_training,
    load_yaml_config,
)


def train_model_artifact(
    project_root: Path,
    artifact_path: Path | None = None,
    holdout_days: int = 180,
    verbose: bool = False,
) -> Path:
    artifact_path = artifact_path or project_root / DEFAULT_MODEL_ARTIFACT_PATH
    progress = ProgressPrinter(enabled=verbose)
    progress.print_header(project_root, artifact_path)

    resources = build_training_resources(project_root, progress=progress.step)

    val_result: Optional[tuple[float, float, int]] = None
    if holdout_days > 0:
        progress.step(f"Evaluating true recent holdout over last {holdout_days} days")
        meta = resources.metadata
        val_result = _run_recent_holdout_validation(
            project_root=project_root,
            holdout_days=holdout_days,
            pretrained_alpha=meta.tuned_alpha if meta is not None else None,
            pretrained_reg_strength=meta.tuned_reg_strength if meta is not None else None,
            pretrained_model_weights=meta.model_weights if meta is not None else None,
        )
        if val_result is not None:
            rps, skill, n = val_result
            progress.step(
                f"Recent {holdout_days}d holdout: RPS={rps:.4f}, skill={skill:+.1%}, n={n:,}"
            )
            if resources.metadata is not None:
                resources.metadata.holdout_rps = rps
                resources.metadata.holdout_skill_score = skill
                resources.metadata.holdout_n_matches = n
                resources.metadata.holdout_days = holdout_days

    progress.step("Saving model artifact")
    save_model_artifact(resources, artifact_path)
    progress.print_summary(resources, val_result)
    return artifact_path


def _run_recent_holdout_validation(
    project_root: Path,
    holdout_days: int = 180,
    pretrained_alpha: float | None = None,
    pretrained_reg_strength: float | None = None,
    pretrained_model_weights: tuple[float, ...] | None = None,
) -> Optional[tuple[float, float, int]]:
    """Train before the recent window, predict held-out matches, and compute RPS."""
    config = load_yaml_config(project_root / "configs" / "model_config.yaml")
    normalizer = TeamNameNormalizer.from_yaml(project_root / "configs" / "team_name_map.yaml")
    results = normalizer.normalize_results(load_results(project_root / "data" / "raw" / "results.csv"))
    results = filter_results_for_training(results, config)
    completed = get_completed_matches(results).sort_values(["date", "match_id"], kind="mergesort")
    if completed.empty:
        return None

    cutoff = pd.Timestamp(
        config.get("training", {}).get("prediction_cutoff_date")
        or completed["date"].max()
    )
    holdout_start = cutoff - pd.Timedelta(days=holdout_days)
    training_results = completed[completed["date"] < holdout_start].copy()
    holdout_matches = completed[
        (completed["date"] >= holdout_start) & (completed["date"] <= cutoff)
    ].copy()

    if training_results.empty or len(holdout_matches) < 10:
        return None

    feature_builder = MatchFeatureBuilder(training_results)
    training_features = feature_builder.build_training_features()
    if training_features.empty:
        return None

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    model_type = str(model_config.get("selected_type", "poisson"))
    max_goals = int(model_config.get("max_goals", 8))
    rho_bounds = model_config.get("rho_bounds", [-0.3, 0.3])

    sample_weight = None
    if training_config.get("use_time_decay", True):
        sample_weight = compute_sample_weights(
            training_features,
            reference_date=holdout_start,
            half_life_days=float(training_config.get("half_life_days", 1460.0)),
            friendly_weight=float(training_config.get("friendly_weight", 0.5)),
        )

    if pretrained_alpha is not None:
        alpha = pretrained_alpha
        reg_strength = pretrained_reg_strength
        model_weights = pretrained_model_weights
    else:
        alpha = float(model_config.get("alpha", 1.0))
        reg_strength = (
            float(model_config["reg_strength"]) if "reg_strength" in model_config else None
        )
        model_weights = _optional_float_tuple(model_config.get("model_weights"))
        if model_config.get("tune_alpha", True):
            tuned = tune_model_hyperparameters(
                training_features=training_features,
                sample_weight=sample_weight,
                candidates=tuple(
                    model_config.get(
                        "alpha_candidates",
                        [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    )
                ),
                max_goals=max_goals,
                model_type=model_type,
                ensemble_weight_candidates=_ensemble_weight_candidates(model_config),
            )
            alpha = tuned.alpha
            reg_strength = tuned.reg_strength if tuned.reg_strength is not None else reg_strength
            model_weights = tuned.model_weights if tuned.model_weights is not None else model_weights

    model = train_score_model(
        training_features,
        model_type=model_type,
        max_goals=max_goals,
        alpha=alpha,
        rho_bounds=(float(rho_bounds[0]), float(rho_bounds[1])),
        sample_weight=sample_weight,
        reg_strength=reg_strength,
        model_weights=model_weights,
    )

    predicted_rows = []
    for match in holdout_matches.itertuples(index=False):
        if match.home_team not in feature_builder.team_universe:
            continue
        if match.away_team not in feature_builder.team_universe:
            continue
        features = feature_builder.build_match_features(
            home_team=str(match.home_team),
            away_team=str(match.away_team),
            venue_mode=_venue_mode_for_match(match),
            prediction_date=match.date,
            tournament=str(match.tournament),
        )
        prediction = model.predict_proba(features).iloc[0]
        predicted_rows.append(
            {
                "prob_home_win": float(prediction["prob_home_win"]),
                "prob_draw": float(prediction["prob_draw"]),
                "home_score": float(match.home_score),
                "away_score": float(match.away_score),
            }
        )

    if len(predicted_rows) < 10:
        return None

    recent = pd.DataFrame(predicted_rows)
    ph = recent["prob_home_win"].to_numpy(dtype=float)
    pd_ = recent["prob_draw"].to_numpy(dtype=float)
    h = recent["home_score"].to_numpy(dtype=float)
    a = recent["away_score"].to_numpy(dtype=float)

    cdf1 = (h > a).astype(float)
    cdf2 = (h >= a).astype(float)

    rps = float((0.5 * ((ph - cdf1) ** 2 + (ph + pd_ - cdf2) ** 2)).mean())
    u_rps = float((0.5 * ((1 / 3 - cdf1) ** 2 + (2 / 3 - cdf2) ** 2)).mean())
    skill = 1.0 - rps / u_rps if u_rps > 0 else 0.0

    return rps, skill, len(recent)


class ProgressPrinter:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.started_at = perf_counter()
        self.last_step_at = self.started_at
        self.step_count = 0

    def print_header(self, project_root: Path, artifact_path: Path) -> None:
        if not self.enabled:
            return
        print("World Cup score model training")
        print(f"Project root: {project_root}")
        print(f"Artifact path: {artifact_path}")
        print("")

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        self.step_count += 1
        print(
            f"[{self.step_count:02d}] {message} "
            f"(+{now - self.last_step_at:.1f}s, total {now - self.started_at:.1f}s)",
            flush=True,
        )
        self.last_step_at = now

    def print_summary(
        self,
        resources: StreamlitAppResources,
        val_result: Optional[tuple[float, float, int]] = None,
    ) -> None:
        if not self.enabled:
            return
        elapsed = perf_counter() - self.started_at
        print("")
        print("Training complete")
        print(f"Teams available : {len(resources.teams):,}")
        if resources.metadata is not None:
            for line in resources.metadata.summary_lines():
                print(line)
        else:
            print(f"Training matches: {resources.training_match_count:,}")
            print(f"Feature cutoff  : {resources.feature_cutoff_date}")
        print(f"Elapsed time    : {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Streamlit POC model artifact.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing configs/ and data/raw/.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=None,
        help="Output model artifact path. Defaults to models/match_score_model.pkl.",
    )
    parser.add_argument(
        "--no-holdout",
        action="store_true",
        help="Skip the recent holdout validation step.",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=180,
        help="Number of recent days used for holdout validation (default: 180).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    artifact_path = train_model_artifact(
        project_root=args.project_root,
        artifact_path=args.artifact_path,
        holdout_days=0 if args.no_holdout else args.holdout_days,
        verbose=not args.quiet,
    )
    print(f"Wrote model artifact: {artifact_path}")


def _venue_mode_for_match(match: object) -> str:
    if bool(match.neutral):
        return "neutral"
    if str(match.country) == str(match.home_team):
        return "team_1_host_advantage"
    if str(match.country) == str(match.away_team):
        return "team_2_host_advantage"
    return "neutral"


def _optional_float_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise ValueError("model_weights must be a list of numbers")
    return tuple(float(item) for item in value)


def _ensemble_weight_candidates(config: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    raw = config.get("ensemble_weight_candidates")
    if raw is None:
        return ((0.25, 0.75), (0.5, 0.5), (0.75, 0.25))
    if not isinstance(raw, list):
        raise ValueError("ensemble_weight_candidates must be a list of pairs")

    candidates = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError("ensemble_weight_candidates must contain weight pairs")
        candidates.append((float(item[0]), float(item[1])))
    return tuple(candidates)


if __name__ == "__main__":
    main()

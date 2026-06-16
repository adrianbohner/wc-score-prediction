from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from wc_predictor.data import (
    DataValidationError,
    TeamNameNormalizer,
    get_completed_matches,
    load_results,
)
from wc_predictor.evaluation.metrics import (
    actual_outcome,
    evaluate_prediction_frame,
    score_probability,
)
from wc_predictor.features import MatchFeatureBuilder
from wc_predictor.features.weights import compute_sample_weights
from wc_predictor.models import train_score_model
from wc_predictor.models.train import tune_model_hyperparameters
from wc_predictor.streamlit_support import load_yaml_config


DEFAULT_WORLD_CUP_YEARS = (2010, 2014, 2018, 2022)


@dataclass(frozen=True)
class BacktestConfig:
    training_start_date: str = "1990-01-01"
    model_type: str = "poisson"
    max_goals: int = 8
    alpha: float = 1.0
    rho_bounds: tuple[float, float] = (-0.3, 0.3)
    half_life_days: float = 1460.0
    friendly_weight: float = 0.5
    use_time_decay: bool = True
    tune_alpha_enabled: bool = True
    alpha_candidates: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    reg_strength: float | None = None
    model_weights: tuple[float, ...] | None = None
    ensemble_weight_candidates: tuple[tuple[float, float], ...] = (
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
    )

    @classmethod
    def from_project_config(cls, config: dict) -> "BacktestConfig":
        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {})
        return cls(
            training_start_date=str(training_cfg.get("start_date", "1990-01-01")),
            model_type=str(model_cfg.get("selected_type", "poisson")),
            max_goals=int(model_cfg.get("max_goals", 8)),
            alpha=float(model_cfg.get("alpha", 1.0)),
            rho_bounds=tuple(model_cfg.get("rho_bounds", [-0.3, 0.3])),
            half_life_days=float(training_cfg.get("half_life_days", 1460.0)),
            friendly_weight=float(training_cfg.get("friendly_weight", 0.5)),
            use_time_decay=bool(training_cfg.get("use_time_decay", True)),
            tune_alpha_enabled=bool(model_cfg.get("tune_alpha", True)),
            alpha_candidates=tuple(
                model_cfg.get("alpha_candidates", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
            ),
            reg_strength=(
                float(model_cfg["reg_strength"]) if "reg_strength" in model_cfg else None
            ),
            model_weights=_optional_float_tuple(model_cfg.get("model_weights")),
            ensemble_weight_candidates=_ensemble_weight_candidates(model_cfg),
        )


@dataclass
class BacktestResult:
    tournament_year: int
    train_until: str | None
    test_matches: int
    predicted_matches: int
    skipped_matches: int
    metrics: dict[str, float]
    model_type: str = "poisson"
    skipped_reason: str | None = None
    tuned_alpha: float | None = None


def run_world_cup_backtests(
    results: pd.DataFrame,
    years: Iterable[int] = DEFAULT_WORLD_CUP_YEARS,
    config: BacktestConfig | None = None,
) -> list[BacktestResult]:
    config = config or BacktestConfig()
    return [run_world_cup_backtest(results, year, config) for year in years]


def run_model_comparison_backtests(
    results: pd.DataFrame,
    years: Iterable[int] = DEFAULT_WORLD_CUP_YEARS,
    model_types: Iterable[str] = ("poisson", "dixon_coles"),
    config: BacktestConfig | None = None,
) -> list[BacktestResult]:
    base_config = config or BacktestConfig()
    all_results: list[BacktestResult] = []
    for model_type in model_types:
        model_config = replace(base_config, model_type=model_type)
        all_results.extend(run_world_cup_backtests(results, years=years, config=model_config))
    return all_results


def run_world_cup_backtest(
    results: pd.DataFrame,
    tournament_year: int,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    completed = get_completed_matches(results).sort_values(["date", "match_id"], kind="mergesort")
    test_matches = _world_cup_matches(completed, tournament_year)
    if test_matches.empty:
        return BacktestResult(
            tournament_year=tournament_year,
            train_until=None,
            test_matches=0,
            predicted_matches=0,
            skipped_matches=0,
            metrics={},
            model_type=config.model_type,
            skipped_reason="No completed FIFA World Cup matches found.",
        )

    tournament_start = pd.Timestamp(test_matches["date"].min())
    train_start = pd.Timestamp(config.training_start_date)
    training_results = completed[
        (completed["date"] >= train_start) & (completed["date"] < tournament_start)
    ].copy()
    if training_results.empty:
        return BacktestResult(
            tournament_year=tournament_year,
            train_until=tournament_start.strftime("%Y-%m-%d"),
            test_matches=len(test_matches),
            predicted_matches=0,
            skipped_matches=len(test_matches),
            metrics={},
            model_type=config.model_type,
            skipped_reason="No training matches available before tournament.",
        )

    feature_builder = MatchFeatureBuilder(training_results)
    training_features = feature_builder.build_training_features()

    sample_weight = None
    if config.use_time_decay:
        sample_weight = compute_sample_weights(
            training_features,
            reference_date=tournament_start,
            half_life_days=config.half_life_days,
            friendly_weight=config.friendly_weight,
        )

    best_alpha = config.alpha
    best_reg_strength = config.reg_strength
    best_model_weights = config.model_weights
    if config.tune_alpha_enabled:
        tuned = tune_model_hyperparameters(
            training_features=training_features,
            sample_weight=sample_weight,
            candidates=config.alpha_candidates,
            max_goals=config.max_goals,
            model_type=config.model_type,
            ensemble_weight_candidates=config.ensemble_weight_candidates,
        )
        best_alpha = tuned.alpha
        best_reg_strength = (
            tuned.reg_strength if tuned.reg_strength is not None else best_reg_strength
        )
        best_model_weights = (
            tuned.model_weights if tuned.model_weights is not None else best_model_weights
        )

    model = train_score_model(
        training_features,
        model_type=config.model_type,
        max_goals=config.max_goals,
        alpha=best_alpha,
        rho_bounds=(float(config.rho_bounds[0]), float(config.rho_bounds[1])),
        sample_weight=sample_weight,
        reg_strength=best_reg_strength,
        model_weights=best_model_weights,
    )

    records = []
    skipped = 0
    for match in test_matches.itertuples(index=False):
        if match.home_team not in feature_builder.team_universe:
            skipped += 1
            continue
        if match.away_team not in feature_builder.team_universe:
            skipped += 1
            continue

        features = feature_builder.build_match_features(
            home_team=str(match.home_team),
            away_team=str(match.away_team),
            venue_mode=_venue_mode_for_match(match),
            prediction_date=tournament_start,
            tournament=str(match.tournament),
        )
        prediction = model.predict_proba(features).iloc[0]
        matrix = model.predict_score_matrix(features)
        records.append(
            {
                "match_id": match.match_id,
                "date": match.date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_home_goals": int(match.home_score),
                "actual_away_goals": int(match.away_score),
                "actual_outcome": actual_outcome(match.home_score, match.away_score),
                "actual_score_prob": score_probability(
                    matrix,
                    match.home_score,
                    match.away_score,
                ),
                "expected_home_goals": float(prediction["expected_home_goals"]),
                "expected_away_goals": float(prediction["expected_away_goals"]),
                "pred_home_goals": int(prediction["pred_home_goals"]),
                "pred_away_goals": int(prediction["pred_away_goals"]),
                "top_scorelines": prediction["top_scorelines"],
                "prob_home_win": float(prediction["prob_home_win"]),
                "prob_draw": float(prediction["prob_draw"]),
                "prob_away_win": float(prediction["prob_away_win"]),
            }
        )

    prediction_frame = pd.DataFrame(records)
    return BacktestResult(
        tournament_year=tournament_year,
        train_until=tournament_start.strftime("%Y-%m-%d"),
        test_matches=len(test_matches),
        predicted_matches=len(prediction_frame),
        skipped_matches=skipped,
        metrics=evaluate_prediction_frame(prediction_frame),
        model_type=config.model_type,
        tuned_alpha=best_alpha if config.tune_alpha_enabled else None,
    )


def build_evaluation_report(results: list[BacktestResult]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "## Selected Model",
        "",
        "The current POC model is selected by `model.selected_type` in `configs/model_config.yaml`.",
        "",
        "Dixon-Coles is available as an enhancement and should replace the baseline only if it validates at least as well.",
        "",
        "## Backtest Summary",
        "",
        "| Tournament | Model | alpha | Train until | Predicted | Skipped | RPS | RPS skill | 1X2 log loss | Brier | Top-5 acc | Draw cal |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        if result.skipped_reason:
            lines.append(
                f"| {result.tournament_year} | {result.model_type} | n/a | n/a"
                f" | 0 | {result.skipped_matches} | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue

        m = result.metrics
        alpha_str = f"{result.tuned_alpha:.4f}" if result.tuned_alpha is not None else "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result.tournament_year),
                    str(result.model_type),
                    alpha_str,
                    str(result.train_until),
                    str(result.predicted_matches),
                    str(result.skipped_matches),
                    _format_metric(m["rps"]),
                    _format_metric(m["rps_skill_score"]),
                    _format_metric(m["one_x_two_log_loss"]),
                    _format_metric(m["brier_score"]),
                    _format_metric(m["top_5_score_accuracy"]),
                    _format_metric(m["draw_calibration"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "* Evaluation uses time-based splits only.",
            "* Predictions are made as pre-tournament predictions using information before the tournament start date.",
            "* **RPS** (Ranked Probability Score) is the primary metric; lower is better.",
            "* **RPS skill** = 1 - RPS_model / RPS_uniform. 0 = no skill vs 1/3-1/3-1/3; higher is better.",
            "* alpha is the L2 regularisation strength selected by time-series CV.",
            "* The report is intended for model comparison, not betting-grade validation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_report(path: Path, results: list[BacktestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_evaluation_report(results), encoding="utf-8")


def run_project_backtests(project_root: Path) -> list[BacktestResult]:
    project_config = load_yaml_config(project_root / "configs" / "model_config.yaml")
    normalizer = TeamNameNormalizer.from_yaml(project_root / "configs" / "team_name_map.yaml")
    results = normalizer.normalize_results(load_results(project_root / "data" / "raw" / "results.csv"))
    return run_world_cup_backtests(
        results,
        years=DEFAULT_WORLD_CUP_YEARS,
        config=BacktestConfig.from_project_config(project_config),
    )


def main() -> None:
    project_root = Path.cwd()
    results = run_project_backtests(project_root)
    report_path = project_root / "outputs" / "evaluation_report.md"
    write_evaluation_report(report_path, results)
    print(f"Wrote {report_path}")


def _world_cup_matches(completed: pd.DataFrame, year: int) -> pd.DataFrame:
    return completed[
        (completed["tournament"] == "FIFA World Cup")
        & (completed["date"].dt.year == year)
    ].copy()


def _venue_mode_for_match(match: object) -> str:
    if bool(match.neutral):
        return "neutral"
    if str(match.country) == str(match.home_team):
        return "team_1_host_advantage"
    if str(match.country) == str(match.away_team):
        return "team_2_host_advantage"
    return "neutral"


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _optional_float_tuple(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise DataValidationError("model_weights must be a list of numbers")
    return tuple(float(item) for item in value)


def _ensemble_weight_candidates(config: dict) -> tuple[tuple[float, float], ...]:
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


if __name__ == "__main__":
    main()

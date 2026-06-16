# World Cup 2026 Score Prediction

Streamlit proof of concept for predicting a single World Cup 2026 match score.

The app will let a user select two teams and receive the most likely scoreline, exact-score probability, confidence hint, alternative scorelines, and win/draw/loss probabilities.

## Requirements

* Python 3.11 or newer
* `data/raw/results.csv`
* Optional for later phases: `data/raw/goalscorers.csv`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run Tests

```powershell
pytest
```

## Train Model Artifact

Train before starting the app:

```powershell
python -m wc_predictor.models.train_app_model
```

This creates:

```text
models/match_score_model.pkl
```

The Streamlit app loads this artifact. Training is intentionally not part of the app UX.

The training command prints progress by default, including row counts, training step names, and elapsed time. To suppress progress output:

```powershell
python -m wc_predictor.models.train_app_model --quiet
```

After `python -m pip install -e ".[dev]"`, you can also run:

```powershell
train-wc-model
```

If you have not installed the package yet, run the module with `PYTHONPATH`:

```powershell
$env:PYTHONPATH='src'
python -m wc_predictor.models.train_app_model
```

## Run Streamlit App

```powershell
streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Generate Evaluation Report

```powershell
$env:PYTHONPATH='src'
python -m wc_predictor.evaluation.backtest
```

## Configuration

Selectable teams live in:

```text
configs/world_cup_2026_teams.yaml
```

After changing this file, retrain the model artifact:

```powershell
python -m wc_predictor.models.train_app_model
```

The active model is configured in:

```text
configs/model_config.yaml
```

Allowed `model.selected_type` values:

```text
poisson
dixon_coles
team_strength
ensemble
```

Use the evaluation report to compare model types before changing the production artifact.

## Manual QA

Use:

```text
docs/manual_qa_checklist.md
```

## Troubleshooting

If imports fail when running a module directly, set:

```powershell
$env:PYTHONPATH='src'
```

If Streamlit startup fails, check:

* `models/match_score_model.pkl` exists
* `data/raw/results.csv` exists
* teams in `configs/world_cup_2026_teams.yaml` exist in the results data
* dependencies were installed with `python -m pip install -e ".[dev]"`

## Current Status

Phases 0 and 1 are implemented: project skeleton, starter configuration, package structure, placeholder Streamlit page, data loading, validation, and smoke tests.

Phase 2 is implemented: config-driven team-name normalization, selectable team loading, team-universe validation, and goalscorer team validation.

Phase 3 is implemented: internal Elo ratings, one-row-per-team match history, rolling form features, attack/defense strength features, venue flags, and match-level feature assembly.

Phase 4 is implemented: simple Poisson baseline training, expected-goals prediction, normalized score matrix generation, top scoreline extraction, and win/draw/loss probability derivation.

Phase 5 is implemented: `predict_match` service layer, confidence labels/explanations, UI-ready response formatting, and validation for same-team, unknown-team, and invalid venue inputs.

Phase 6 is implemented: Streamlit UI for team selection, venue selection, prediction display, alternative scorelines, outcome probabilities, loading states, and friendly errors.

Phase 7 is implemented: evaluation metrics, historical World Cup backtest runner, markdown report writer, and starter `outputs/evaluation_report.md`.

Phase 8 is implemented: Dixon-Coles low-score adjustment, bounded `rho` optimization, model training dispatcher, and comparison-ready backtests. The app uses `model.selected_type` from `configs/model_config.yaml`.

Phase 9 is implemented: pre-trained model artifact workflow, handoff documentation, manual QA checklist, final run commands, and verification notes.

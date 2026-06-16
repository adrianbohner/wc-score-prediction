# Manual QA Checklist

Use this checklist before handing the POC to someone else.

## Setup

1. Create and activate a virtual environment.
2. Install the project with dev dependencies.
3. Confirm `data/raw/results.csv` exists.
4. Confirm `configs/world_cup_2026_teams.yaml` contains teams that exist in the historical results data.

## Automated Checks

Run:

```powershell
pytest
```

Expected result:

```text
all tests pass
```

## Train Model Artifact

Run:

```powershell
python -m wc_predictor.models.train_app_model
```

Alternative after editable install:

```powershell
train-wc-model
```

Check:

1. `models/match_score_model.pkl` exists.
2. The command prints training progress.
3. The command prints the artifact path.

## Streamlit Smoke Test

Run:

```powershell
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

Check:

1. Page title is `World Cup 2026 Score Prediction`.
2. Team 1 and Team 2 selectors appear.
3. Venue selector appears.
4. Predict button is disabled or blocked when the same team is selected.
5. Predict button works for two different teams.
6. Result shows `Most likely score`.
7. Result shows `Chance of this exact score`.
8. Result shows `Confidence`.
9. Result shows `Other possible scores`.
10. Result shows `Match outcome chances`.
11. Result includes `Predictions are probabilities, not guarantees.`
12. `About the model` expands and shows training match count and feature cutoff date.

## Evaluation Report

Run:

```powershell
$env:PYTHONPATH='src'
python -m wc_predictor.evaluation.backtest
```

Check:

1. `outputs/evaluation_report.md` is created or updated.
2. The report includes a backtest summary table.
3. The selected model is listed.

## Model Selection

The app uses:

```yaml
model:
  selected_type: "poisson"
```

Allowed values:

```text
poisson
dixon_coles
```

Keep `poisson` as the default unless evaluation shows `dixon_coles` is at least as reliable.

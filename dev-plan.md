# Development Plan: World Cup 2026 Match Score Prediction POC

## 1. Goal

Build a Streamlit proof of concept that lets a non-technical user select two World Cup 2026 teams and receive:

* the most likely final score
* the probability of that exact score
* a simple confidence label
* 3 to 5 alternative plausible scorelines
* win/draw/loss probabilities for context

The POC must stay focused on match prediction only. Group simulation, knockout prediction, standings, saved user predictions, and external data are not part of this plan.

---

## 2. Delivery Strategy

Use a staged approach:

1. Build a reliable simple Poisson prediction pipeline first.
2. Put the prediction behind a clean `predict_match` interface.
3. Build the Streamlit UI on top of that interface.
4. Add evaluation and diagnostics.
5. Add Dixon-Coles only after the baseline is working and testable.

This keeps the POC usable early while preserving a clear path to the stronger model.

---

## 3. Proposed Project Structure

```text
wc-score-prediction/
  data/
    raw/
      results.csv
      goalscorers.csv
    processed/
  configs/
    model_config.yaml
    team_name_map.yaml
    world_cup_2026_teams.yaml
  models/
    match_score_model.pkl
  outputs/
    evaluation_report.md
  src/
    wc_predictor/
      __init__.py
      data/
        load_data.py
        validate_data.py
        normalize_teams.py
      features/
        elo.py
        rolling_features.py
        feature_builder.py
      models/
        poisson_baseline.py
        dixon_coles.py
        train.py
        predict.py
      presentation/
        confidence.py
        formatting.py
      evaluation/
        metrics.py
        backtest.py
  app/
    streamlit_app.py
  tests/
    test_data_validation.py
    test_elo.py
    test_features.py
    test_score_matrix.py
    test_prediction_response.py
  pyproject.toml
  README.md
```

For the POC, `app/streamlit_app.py` should call the same prediction interface used by tests and scripts:

```python
predict_match(home_team, away_team, venue_mode="neutral", prediction_date=None)
```

---

## 4. Phase 0: Project Bootstrap

### Tasks

* Create project folders.
* Add `pyproject.toml` with core dependencies:
  * `pandas`
  * `numpy`
  * `scipy`
  * `scikit-learn`
  * `pyyaml`
  * `streamlit`
  * `pytest`
* Add initial configs:
  * `configs/model_config.yaml`
  * `configs/team_name_map.yaml`
  * `configs/world_cup_2026_teams.yaml`
* Add a short `README.md` with local run commands.

### Deliverables

* Runnable Python project skeleton.
* Streamlit dependency included.
* Empty or starter config files committed.

### Acceptance Checks

* `pytest` starts without import errors.
* `streamlit run app/streamlit_app.py` can start a placeholder page.

---

## 5. Phase 1: Data Loading and Validation

### Tasks

* Implement `load_results(path)`.
* Implement `load_goalscorers(path)` as optional.
* Parse date columns.
* Validate required columns.
* Validate non-negative integer scores for completed matches.
* Create a completed-match filter.
* Generate a stable `match_id`.

### Deliverables

* `src/wc_predictor/data/load_data.py`
* `src/wc_predictor/data/validate_data.py`
* data validation tests

### Acceptance Checks

* Missing required columns raise clear errors.
* Invalid dates raise clear errors.
* Negative scores raise clear errors.
* Completed matches can be loaded into a clean DataFrame.

---

## 6. Phase 2: Team Normalization and Selectable Team List

### Tasks

* Implement team-name normalization from `configs/team_name_map.yaml`.
* Apply normalization to `home_team`, `away_team`, and scorer `team`.
* Validate that UI-selectable teams exist in the historical result data.
* Sort UI teams alphabetically.
* Start with a manually maintained `world_cup_2026_teams.yaml`.

### Deliverables

* `src/wc_predictor/data/normalize_teams.py`
* `configs/team_name_map.yaml`
* `configs/world_cup_2026_teams.yaml`
* normalization tests

### Acceptance Checks

* Same mapping is applied consistently across both CSVs.
* Unknown UI teams are reported before app startup.
* Same-team matchup can be rejected by the prediction layer.

---

## 7. Phase 3: Feature Pipeline

### Tasks

* Implement internal Elo calculation from `results.csv`.
* Build one-row-per-team-per-match table.
* Implement rolling features for 5-match and 10-match windows.
* Implement attack and defense strength features.
* Implement neutral/host venue feature flags.
* Implement one function that can build features for:
  * historical training rows
  * a user-selected future matchup

### Required Features

| Feature Group | Required Fields |
| --- | --- |
| Elo | `home_elo_pre`, `away_elo_pre`, `elo_diff`, `elo_abs_diff` |
| Form | `points_avg_5`, `points_avg_10`, `goal_diff_avg_10` |
| Goals | `goals_for_avg_5`, `goals_for_avg_10`, `goals_against_avg_5`, `goals_against_avg_10` |
| Strength | `attack_strength_10`, `defense_strength_10` |
| History | `matches_available_10`, low-history flags |
| Venue | neutral, team-1 host advantage, team-2 host advantage |

### Deliverables

* `src/wc_predictor/features/elo.py`
* `src/wc_predictor/features/rolling_features.py`
* `src/wc_predictor/features/feature_builder.py`
* feature tests

### Acceptance Checks

* Current match is excluded from its own feature calculations.
* Same-date matches are excluded from as-of-date features.
* Any valid UI matchup can produce one feature row.
* Missing team history receives deterministic defaults and warning flags.

---

## 8. Phase 4: Simple Poisson Baseline

### Tasks

* Train a regularized Poisson-style goal model or practical baseline using engineered features.
* Predict expected goals for team 1 and team 2.
* Generate a scoreline probability matrix from `0-0` through `max_goals-max_goals`.
* Normalize the score matrix.
* Derive:
  * top scoreline
  * top 5 scorelines
  * home/team-1 win probability
  * draw probability
  * away/team-2 win probability

### Deliverables

* `src/wc_predictor/models/poisson_baseline.py`
* `src/wc_predictor/models/train.py`
* score matrix tests

### Acceptance Checks

* Expected goals are always positive.
* Score matrix probabilities are non-negative.
* Score matrix sums to 1.
* Win/draw/loss probabilities sum to 1.
* Top scoreline matches the highest probability cell.

---

## 9. Phase 5: Prediction Response Layer

### Tasks

* Implement `predict_match`.
* Implement confidence label thresholds:
  * `< 0.12`: Low
  * `0.12 - 0.18`: Medium
  * `> 0.18`: High
* Implement confidence explanation text.
* Format top scorelines as UI-ready objects.
* Add validation for:
  * unknown teams
  * same team selected twice
  * invalid venue mode

### Deliverables

* `src/wc_predictor/models/predict.py`
* `src/wc_predictor/presentation/confidence.py`
* `src/wc_predictor/presentation/formatting.py`
* prediction response tests

### Acceptance Checks

* Response includes every field required by `dev-spec.md`.
* Alternatives are sorted by probability descending.
* Confidence label follows config thresholds.
* Errors are plain enough to display directly in Streamlit.

---

## 10. Phase 6: Streamlit POC UI

### Tasks

* Build `app/streamlit_app.py`.
* Load selectable teams from config.
* Use two searchable `st.selectbox` controls.
* Add a simple venue selector:
  * `Neutral`
  * `Team 1 host advantage`
  * `Team 2 host advantage`
* Disable or block prediction until two different teams are selected.
* Add a `Predict` button.
* Display:
  * large main score
  * exact-score probability
  * confidence label and explanation
  * top 5 scorelines
  * match outcome probabilities
  * low-history warning when applicable
* Add an optional collapsed `About the model` section.

### POC Layout

```text
Title
Team 1 selector | Team 2 selector | Venue selector
Predict button

Most likely score
Chance of this exact score
Confidence

Other possible scores
Match outcome chances
```

### UI Copy

Use:

* `Most likely score`
* `Chance of this exact score`
* `Other possible scores`
* `Match outcome chances`
* `Predictions are probabilities, not guarantees.`

Avoid on the main screen:

* `Poisson`
* `Dixon-Coles`
* `lambda`
* `feature vector`

### Deliverables

* `app/streamlit_app.py`
* Streamlit smoke test notes in `README.md`

### Acceptance Checks

* A non-technical user can make a prediction from the first screen.
* Same-team selection shows a friendly warning.
* Prediction result is readable without scrolling on a typical laptop screen.
* The UI does not require users to understand the model.

---

## 11. Phase 7: Evaluation Report

### Tasks

* Implement backtests for historical World Cups where data allows.
* Compare simple Poisson baseline against Dixon-Coles once available.
* Calculate:
  * exact score log loss
  * home goals MAE
  * away goals MAE
  * exact score accuracy
  * top-5 score accuracy
  * 1X2 log loss
  * Brier score
* Generate `outputs/evaluation_report.md`.

### Deliverables

* `src/wc_predictor/evaluation/metrics.py`
* `src/wc_predictor/evaluation/backtest.py`
* `outputs/evaluation_report.md`

### Acceptance Checks

* Evaluation uses time-based splits only.
* Report states which model is selected for the POC.
* Report includes limitations in plain language.

---

## 12. Phase 8: Dixon-Coles Enhancement

### Tasks

* Implement Dixon-Coles low-score adjustment.
* Add optimization with parameter bounds.
* Compare against the simple Poisson baseline.
* Keep simple Poisson as fallback.
* Use Dixon-Coles only if validation is stable and at least as good as baseline.

### Deliverables

* `src/wc_predictor/models/dixon_coles.py`
* model comparison in evaluation report
* config flag for selected model type

### Acceptance Checks

* Dixon-Coles score matrix sums to 1 after normalization.
* Low-score probabilities are adjusted only in supported score cells.
* Model selection decision is documented.
* Streamlit UI behavior is unchanged regardless of selected model.

---

## 13. Phase 9: POC Polish and Handoff

### Tasks

* Improve README with setup, training, test, and Streamlit run commands.
* Add a short model limitation note in the UI.
* Confirm all tests pass.
* Confirm Streamlit starts locally.
* Add a small manual QA checklist.

### Deliverables

* final README instructions
* passing tests
* local Streamlit app

### Acceptance Checks

* Fresh setup instructions are clear.
* User can run:

```text
streamlit run app/streamlit_app.py
```

* App loads team selectors and can produce a prediction.

---

## 14. POC Priority Order

Build the first usable POC in this order:

```text
1. Project bootstrap
2. Data loading and validation
3. Team normalization
4. Simple feature builder
5. Simple Poisson baseline
6. Prediction response formatter
7. Streamlit UI
8. Unit tests around prediction correctness
9. Evaluation report
10. Dixon-Coles enhancement
```

The key checkpoint is after step 7: a user should already be able to select two teams and get a reasonable prediction, even if the model is still the simple baseline.

---

## 15. Definition of Done

The POC is done when:

* `results.csv` can be loaded and validated.
* The app has a selectable list of World Cup 2026 teams.
* A user can select two different teams in Streamlit.
* The app returns a most likely score in `Team 1 goals - Team 2 goals` format.
* The app shows the exact-score probability.
* The app shows Low/Medium/High confidence.
* The app shows 3 to 5 alternative scorelines.
* The app shows win/draw/loss probabilities.
* The prediction pipeline has unit tests for the score matrix and response format.
* The README explains how to run the Streamlit app locally.

---

## 16. Explicit Non-Goals

Do not build these for the POC:

* group-stage simulation
* knockout bracket
* tournament winner model
* live data ingestion
* betting recommendation logic
* user login
* prediction history database
* complex dashboard UI
* external FIFA ranking or Elo integration


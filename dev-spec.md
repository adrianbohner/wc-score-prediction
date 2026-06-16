# Development Specification: World Cup 2026 Match Score Prediction App

## 1. Objective

Build a simple prediction application for the FIFA World Cup 2026.

The application must answer one user question:

```text
If Team A plays Team B, what is the most likely final score?
```

The first version only needs match-level score prediction:

```text
home_goals - away_goals
```

It does not need group-table simulation, knockout simulation, tournament winner prediction, player-level prediction, or live match updates.

The app should be easy enough for non-technical users to use. A user selects two teams in the UI, clicks predict, and receives:

* the most likely scoreline
* the probability of that exact scoreline
* a simple confidence hint
* a few other plausible scorelines
* optional win/draw/loss probabilities for context

The selected modelling approach remains a lightweight football score model:

```text
Dixon-Coles adjusted Poisson model, with a simple Poisson fallback
```

The model must use only the available project datasets unless the spec is explicitly changed later:

1. `results.csv`
2. `goalscorers.csv`

No external data such as FIFA rankings, Elo websites, betting odds, squad values, injuries, xG, or lineups should be required for Version 1.

---

## 2. Product Scope

## 2.1 In Scope

Version 1 includes:

* loading and validating `results.csv`
* loading `goalscorers.csv` as optional supplementary data
* team name normalization
* model training from historical international results
* internal team-strength features derived from `results.csv`
* rolling team form features
* attack and defense strength features
* venue/host/neutral features when available
* optional basic scorer-profile features from `goalscorers.csv`
* scoreline probability generation
* one simple UI for selecting two World Cup 2026 teams
* output of the best score prediction and several alternatives
* a confidence/probability explanation suitable for casual users
* basic model evaluation on historical World Cup matches

## 2.2 Out of Scope

Version 1 excludes:

* group-stage simulation
* knockout simulation
* full tournament bracket prediction
* champion prediction
* standings table prediction
* live match updates
* live API integration
* external Elo ratings
* FIFA rankings
* xG, shots, possession, or event data
* betting odds
* squad lists
* injuries or suspensions
* player availability
* manual expert overrides
* user accounts
* saved user predictions
* multi-language UI

---

## 3. Users and Main Workflow

## 3.1 Target User

The target user is a football fan or analyst who wants to try the prediction methodology without writing code.

The user should not need to understand model internals, CSV files, notebooks, or command-line tools.

## 3.2 Primary UI Flow

1. User opens the app.
2. User selects a home/team-1 country from a dropdown.
3. User selects an away/team-2 country from a dropdown.
4. User optionally chooses venue context if available:
   * neutral venue
   * home-team host advantage
   * away-team host advantage
5. User clicks `Predict`.
6. App displays the predicted score and confidence information.

## 3.3 Required UI Output

For one selected matchup, show:

| UI Element | Description |
| --- | --- |
| Main prediction | Most likely scoreline, for example `Brazil 2 - 1 Germany` |
| Exact-score probability | Probability of the displayed exact scoreline |
| Confidence hint | Human-readable certainty label |
| Alternative scorelines | Next best 3 to 5 scorelines |
| Win/draw/loss probabilities | Context probabilities derived from the score matrix |
| Expected goals | Optional compact display of expected goals for each team |
| Model note | Short text that predictions are probabilistic, not guaranteed |

The UI should be simple and direct. Avoid dashboards, complex filters, charts, and technical model explanations in the first screen.

---

## 4. Available Input Data

## 4.1 `results.csv`

### Columns

| Column | Type | Description |
| --- | --- | --- |
| `date` | string/date | Match date |
| `home_team` | string | Home team name |
| `away_team` | string | Away team name |
| `home_score` | float | Home team goals |
| `away_score` | float | Away team goals |
| `tournament` | string | Tournament name |
| `city` | string | Match city |
| `country` | string | Match country |
| `neutral` | boolean | Whether the match was played at a neutral venue |

### Usage

This is the main modelling dataset. Rows with known `home_score` and `away_score` are used for model training and validation.

For Version 1, future fixture rows are not required. The UI can generate prediction rows dynamically from the two teams selected by the user.

### Key Rules

* Use only rows with known `home_score` and `away_score` for training.
* Do not use rows from the future relative to the prediction cutoff date.
* For pre-match features, only information available before the match date may be used.
* Because dates have no kickoff time, do not use same-date matches when calculating features.

## 4.2 `goalscorers.csv`

### Columns

| Column | Type | Description |
| --- | --- | --- |
| `date` | string/date | Match date |
| `home_team` | string | Home team name |
| `away_team` | string | Away team name |
| `team` | string | Team that scored |
| `scorer` | string | Goal scorer name |
| `minute` | float | Goal minute |
| `own_goal` | boolean | Whether the goal was an own goal |
| `penalty` | boolean | Whether the goal was a penalty |

### Usage

Use this dataset only for optional lightweight team scoring-profile features.

It must not be treated as complete event data. It does not include shots, xG, assists, lineups, minutes played, or player availability.

### Key Rules

* Scorer-derived features are supplementary, not mandatory for the UI.
* Every scorer-derived feature must be calculated using matches before the prediction date.
* Add missingness or coverage indicators because scorer data may be incomplete.
* Do not use scorer rows from a target match itself.

---

## 5. Prediction Targets and Outputs

## 5.1 Primary Prediction Target

The model predicts the final score:

```text
home_goals = home_score
away_goals = away_score
```

The main user-facing prediction is:

```text
home_team home_goals - away_goals away_team
```

Example:

```text
France 2 - 1 Argentina
```

## 5.2 Derived Probabilities

From the scoreline probability matrix, derive:

```text
P(home win)
P(draw)
P(away win)
P(each exact scoreline)
```

## 5.3 Required Prediction Response

The prediction service should return:

| Field | Description |
| --- | --- |
| `home_team` | Selected home/team-1 country |
| `away_team` | Selected away/team-2 country |
| `pred_home_goals` | Goal count from the most likely scoreline |
| `pred_away_goals` | Goal count from the most likely scoreline |
| `most_likely_score` | String or object for the top scoreline |
| `most_likely_score_prob` | Probability of that exact scoreline |
| `confidence_label` | Simple confidence label |
| `confidence_explanation` | One short sentence for the UI |
| `top_scorelines` | Top 3 to 5 scoreline alternatives |
| `prob_home_win` | Probability home/team-1 wins |
| `prob_draw` | Probability of draw |
| `prob_away_win` | Probability away/team-2 wins |
| `expected_home_goals` | Model expected goals for home/team-1 |
| `expected_away_goals` | Model expected goals for away/team-2 |
| `model_version` | Model version identifier |
| `feature_cutoff_date` | Date up to which features were calculated |

Example response:

```json
{
  "home_team": "France",
  "away_team": "Argentina",
  "pred_home_goals": 1,
  "pred_away_goals": 1,
  "most_likely_score": "France 1 - 1 Argentina",
  "most_likely_score_prob": 0.118,
  "confidence_label": "Low",
  "confidence_explanation": "Football scores are spread across many close outcomes, so the top exact score is only a modest favorite.",
  "top_scorelines": [
    {"score": "1-1", "probability": 0.118},
    {"score": "1-0", "probability": 0.104},
    {"score": "0-1", "probability": 0.098},
    {"score": "2-1", "probability": 0.083},
    {"score": "0-0", "probability": 0.079}
  ],
  "prob_home_win": 0.38,
  "prob_draw": 0.29,
  "prob_away_win": 0.33,
  "expected_home_goals": 1.28,
  "expected_away_goals": 1.17,
  "model_version": "v1.0.0",
  "feature_cutoff_date": "2026-06-01"
}
```

---

## 6. Confidence Hint Specification

Exact football score prediction is inherently uncertain. The UI must not overstate certainty.

## 6.1 Confidence Labels

Use the probability of the most likely exact scoreline as the primary exact-score confidence signal:

| Top exact-score probability | Label |
| ---: | --- |
| `< 0.12` | Low |
| `0.12 - 0.18` | Medium |
| `> 0.18` | High |

These thresholds can be adjusted after validation.

## 6.2 Confidence Explanation

The UI should include one short explanation:

* Low: `Several scorelines are similarly likely. Treat this as a weak favorite.`
* Medium: `This is the clearest scoreline, but nearby scores remain plausible.`
* High: `The model sees this as a relatively strong exact-score pick.`

## 6.3 Alternative Scorelines

Always show at least 3 alternative scorelines. Recommended default:

```text
top 5 scorelines by probability
```

The top scoreline should also appear in the alternatives list, marked as the main prediction.

---

## 7. Modelling Approach

## 7.1 Selected Model

Use a Dixon-Coles adjusted Poisson model when it performs at least as well as a simple Poisson model in validation.

The model estimates goal intensities:

```text
lambda_home = expected goals for home/team-1
lambda_away = expected goals for away/team-2
```

Then it calculates:

```text
P(home_goals = x, away_goals = y)
```

for all scorelines from `0-0` through a configured maximum goal count.

Recommended:

```text
max_goals = 8
```

Normalize the matrix so all scoreline probabilities sum to 1.

## 7.2 Simple Poisson Fallback

If Dixon-Coles is unstable or does not improve validation performance, ship a regularized independent Poisson model first.

The UI output should be identical for both model types.

## 7.3 Model Formula

The base model should estimate:

```text
log(lambda_home) =
    intercept
    + home_attack_strength
    - away_defense_strength
    + venue_or_host_adjustment
    + beta_features_home
```

```text
log(lambda_away) =
    intercept
    + away_attack_strength
    - home_defense_strength
    + venue_or_host_adjustment
    + beta_features_away
```

For neutral matches:

```text
home_advantage = 0
```

For World Cup 2026 host-country matches, United States, Mexico, or Canada may receive host advantage only when the selected venue/country context supports it.

---

## 8. Feature Engineering

All features must use as-of-date logic.

For a match played or predicted on date `D`, use only matches where:

```text
historical_match.date < D
```

## 8.1 Team Name Normalization

Requirements:

* Strip leading/trailing spaces.
* Apply the same team-name mapping to both CSV files.
* Keep the mapping in a config file.
* Validate that all UI-selectable teams exist in the normalized training data.

Example config:

```yaml
United States: United States
USA: United States
Czechia: Czech Republic
Czech Republic: Czech Republic
Curacao: Curacao
```

## 8.2 Internal Elo-Style Rating

Create an internal Elo rating from `results.csv`.

Initial rating:

```text
1500
```

Features:

| Feature | Description |
| --- | --- |
| `home_elo_pre` | Home/team-1 Elo before match |
| `away_elo_pre` | Away/team-2 Elo before match |
| `elo_diff` | `home_elo_pre - away_elo_pre` |
| `elo_abs_diff` | Absolute Elo difference |

Use only pre-match Elo values for training rows.

## 8.3 Rolling Form Features

Recommended windows:

```text
5 matches
10 matches
```

Required features per team:

| Feature | Description |
| --- | --- |
| `points_avg_5` | Average points in previous 5 matches |
| `points_avg_10` | Average points in previous 10 matches |
| `goals_for_avg_5` | Average goals scored in previous 5 matches |
| `goals_for_avg_10` | Average goals scored in previous 10 matches |
| `goals_against_avg_5` | Average goals conceded in previous 5 matches |
| `goals_against_avg_10` | Average goals conceded in previous 10 matches |
| `goal_diff_avg_10` | Average goal difference in previous 10 matches |
| `matches_available_10` | Number of available matches in 10-match window |

## 8.4 Attack and Defense Strength

Use actual goals, not xG.

```text
attack_strength_10 = team_goals_for_avg_10 / global_goals_per_team_match
defense_strength_10 = team_goals_against_avg_10 / global_goals_against_per_team_match
```

Interpretation:

* attack strength above 1 means above-average scoring
* defense strength below 1 means better-than-average defense
* defense strength above 1 means weaker defense

## 8.5 Venue and Host Features

Version 1 UI should support a simple venue mode:

| UI value | Feature behavior |
| --- | --- |
| `Neutral` | No home advantage |
| `Team 1 host advantage` | Apply host/home adjustment to selected team 1 |
| `Team 2 host advantage` | Apply host/home adjustment to selected team 2 |

If the app later predicts official fixtures with known city/country, derive this automatically from `country`, `home_team`, `away_team`, and `neutral`.

## 8.6 Optional Scorer Features

Use `goalscorers.csv` only if coverage is good enough.

Possible features:

| Feature | Description |
| --- | --- |
| `penalty_goal_share` | Share of goals from penalties |
| `own_goal_for_share` | Share of goals credited as opponent own goals |
| `top_scorer_goal_share` | Concentration of goals in top scorer |
| `scorer_coverage_flag` | Whether scorer data exists for the team/window |

If these features create instability or many missing values, exclude them from Version 1.

---

## 9. Data Pipeline

## 9.1 Pipeline Overview

```text
1. Load results.csv
2. Load goalscorers.csv if available
3. Normalize team names
4. Validate required columns and score values
5. Build historical completed-match table
6. Build as-of-date team features
7. Train baseline Poisson model
8. Train Dixon-Coles model if stable
9. Validate models on historical World Cup matches
10. Save the selected model artifact
11. Serve predictions through API/UI
```

## 9.2 Data Validation Rules

For `results.csv`:

* required columns must exist
* dates must parse successfully
* completed matches must have non-negative integer scores
* team names must be non-empty
* duplicate match IDs should be reported

For `goalscorers.csv`:

* required columns must exist if scorer features are enabled
* dates must parse successfully
* scorer team must exist in normalized team universe
* own-goal and penalty fields must be boolean-compatible

---

## 10. Prediction Service

## 10.1 API Contract

Expose one prediction function or endpoint:

```text
predict_match(home_team, away_team, venue_mode, prediction_date)
```

Recommended HTTP endpoint:

```text
POST /api/predict
```

Request:

```json
{
  "home_team": "France",
  "away_team": "Argentina",
  "venue_mode": "neutral",
  "prediction_date": "2026-06-01"
}
```

Response should match the required prediction response in Section 5.3.

## 10.2 Validation

The service must reject:

* unknown teams
* same team selected twice
* invalid venue mode
* prediction dates before either team has enough history

Use user-friendly error messages in the UI.

---

## 11. UI Specification

## 11.1 Design Goal

The UI should be extremely simple:

```text
Select two teams -> click Predict -> read prediction
```

No technical settings should be required for normal users.

## 11.2 First Screen

Required controls:

| Control | Requirement |
| --- | --- |
| Team 1 dropdown | Searchable or alphabetized list of World Cup 2026 teams |
| Team 2 dropdown | Searchable or alphabetized list of World Cup 2026 teams |
| Venue selector | Default to `Neutral` |
| Predict button | Disabled until two different teams are selected |

Required result display:

| Area | Requirement |
| --- | --- |
| Main score | Large, clear predicted score |
| Probability | Show exact-score probability as a percentage |
| Confidence label | Low/Medium/High |
| Alternatives | Compact list of top 3 to 5 scorelines |
| Outcome probabilities | Home/team-1 win, draw, away/team-2 win |

## 11.3 UI Copy

Use plain language:

* `Most likely score`
* `Chance of this exact score`
* `Other possible scores`
* `Match outcome chances`

Avoid technical wording in the main UI:

* Poisson
* Dixon-Coles
* lambda
* calibration
* feature vector

Technical details can be placed in an optional `About the model` section.

## 11.4 Empty and Error States

Required states:

* initial state before teams are selected
* loading state while prediction is running
* validation error when same team is selected twice
* service/model error if prediction cannot be generated
* low-history warning if one team has limited historical data

---

## 12. Evaluation Strategy

## 12.1 Validation Method

Use time-based validation. Do not use random train/test split.

Recommended backtests:

| Train until | Test tournament |
| --- | --- |
| before 2010 World Cup | 2010 World Cup |
| before 2014 World Cup | 2014 World Cup |
| before 2018 World Cup | 2018 World Cup |
| before 2022 World Cup | 2022 World Cup |

For each backtest:

1. Train using matches before the tournament.
2. Predict World Cup matches.
3. Compare predicted score probabilities with actual final scores.

## 12.2 Metrics

Primary metrics:

| Metric | Description |
| --- | --- |
| exact score log loss | Main quality metric for scoreline probabilities |
| home goals MAE | Mean absolute error for home/team-1 goals |
| away goals MAE | Mean absolute error for away/team-2 goals |
| exact score accuracy | Share of matches where top scoreline is exact |
| top-5 score accuracy | Share of matches where actual score appears in alternatives |

Context metrics:

| Metric | Description |
| --- | --- |
| 1X2 log loss | Quality of win/draw/loss probabilities |
| Brier score | Probability calibration |
| draw calibration | Important because football draws are difficult |

## 12.3 Success Criterion

Ship the simplest model that provides stable and calibrated scoreline probabilities.

The Dixon-Coles model should be used only if it improves or matches the simple Poisson baseline on:

* exact score log loss
* top-5 score accuracy
* 1X2 log loss
* draw calibration

---

## 13. Software Architecture

## 13.1 Suggested Project Structure

```text
worldcup_prediction/
  data/
    raw/
      results.csv
      goalscorers.csv
    processed/
  configs/
    team_name_map.yaml
    model_config.yaml
    world_cup_2026_teams.yaml
  src/
    data/
      load_data.py
      validate_data.py
      normalize_teams.py
    features/
      elo.py
      rolling_features.py
      scorer_features.py
      feature_builder.py
    models/
      poisson_baseline.py
      dixon_coles.py
      train.py
      predict.py
    app/
      api.py
      ui.py
    evaluation/
      metrics.py
      backtest.py
    utils/
      dates.py
      io.py
  tests/
    test_data_validation.py
    test_elo.py
    test_features.py
    test_prediction_response.py
    test_ui_validation.py
  pyproject.toml
  README.md
```

The exact UI framework is flexible. A Streamlit app is acceptable for the first version because it supports a simple interactive UI with minimal overhead. A FastAPI plus frontend approach is also acceptable if the project needs a more app-like deployment.

## 13.2 Main Components

### Data Loader

Responsibilities:

* read CSV files
* parse dates
* enforce column types
* return clean raw DataFrames

### Team Normalizer

Responsibilities:

* apply team-name mapping
* validate teams
* report unmatched names

### Feature Builder

Responsibilities:

* create as-of-date features for training matches
* create feature row for any user-selected matchup
* handle missing history consistently

### Score Model

Responsibilities:

* train Poisson or Dixon-Coles model
* predict expected goals
* produce scoreline probability matrix
* derive top scorelines and outcome probabilities

### Prediction Formatter

Responsibilities:

* choose the most likely scoreline
* assign confidence label
* format alternatives
* return UI-ready response object

### UI App

Responsibilities:

* show team selectors
* validate user input
* call prediction service
* display result clearly

---

## 14. Configuration

Example `model_config.yaml`:

```yaml
training:
  start_date: "1990-01-01"
  prediction_cutoff_date: "2026-06-01"
  use_time_decay: true
  half_life_days: 1460

elo:
  initial_rating: 1500
  base_k: 30
  min_k: 8
  max_k: 40
  home_advantage: 0

features:
  rolling_windows: [5, 10]
  include_scorer_features: false

model:
  preferred_type: "dixon_coles"
  fallback_type: "poisson"
  max_goals: 8
  rho_bounds: [-0.3, 0.3]

ui:
  default_venue_mode: "neutral"
  alternative_scoreline_count: 5
  confidence_thresholds:
    low_max: 0.12
    medium_max: 0.18
```

Example `world_cup_2026_teams.yaml`:

```yaml
teams:
  - Argentina
  - Brazil
  - Canada
  - France
  - Germany
  - Mexico
  - Spain
  - United States
```

This file should contain only teams that users are allowed to select in the UI. Update it when the final World Cup 2026 team list is known.

---

## 15. Output Artifacts

## 15.1 Model Artifact

```text
models/match_score_model.pkl
```

Contains:

* trained model parameters
* team mappings
* feature list
* config
* training metadata
* model version

## 15.2 Evaluation Report

```text
outputs/evaluation_report.md
```

Should include:

* selected model type
* baseline comparison
* exact score metrics
* top-5 scoreline accuracy
* 1X2 probability metrics
* calibration notes
* known limitations

## 15.3 Optional Batch Prediction Output

The UI is the primary delivery mechanism, but a batch output can be useful for testing:

```text
outputs/world_cup_2026_match_predictions.csv
```

Suggested columns:

| Column |
| --- |
| `home_team` |
| `away_team` |
| `pred_home_goals` |
| `pred_away_goals` |
| `most_likely_score_prob` |
| `confidence_label` |
| `top_scorelines_json` |
| `prob_home_win` |
| `prob_draw` |
| `prob_away_win` |
| `expected_home_goals` |
| `expected_away_goals` |
| `model_version` |
| `feature_cutoff_date` |

---

## 16. Testing Requirements

## 16.1 Unit Tests

### Data Validation

Test that:

* missing columns raise errors
* invalid dates raise errors
* negative scores raise errors
* duplicated match IDs are detected or reported

### Elo

Test that:

* Elo before the first match equals initial rating
* winner gains points
* loser loses points
* draw updates are symmetrical
* ratings are stored before the current match update

### Features

Test that:

* current match is excluded from rolling calculations
* same-date matches are excluded
* first matches receive default values
* missing history is handled consistently

### Score Model

Test that:

* score matrix sums to 1 after normalization
* probabilities are non-negative
* win/draw/loss probabilities sum to 1
* expected goals are positive
* top scoreline matches the maximum matrix probability

### Prediction Response

Test that:

* response includes all UI-required fields
* same-team matchup is rejected
* unknown team is rejected
* confidence label follows configured thresholds
* alternatives are sorted by probability descending

### UI

Test that:

* predict button is disabled until two different teams are selected
* loading state is shown while prediction runs
* errors are shown in plain language
* result display includes main score, probability, confidence, alternatives, and outcome probabilities

---

## 17. Development Milestones

## Milestone 1: Data Foundation

Deliverables:

* data loader
* validation report
* team normalization
* completed-match training dataset

Acceptance criteria:

* CSV files load successfully
* completed matches are identified
* team names are normalized consistently

## Milestone 2: Feature Pipeline

Deliverables:

* internal Elo features
* rolling form features
* attack/defense features
* optional scorer features
* feature row generation for arbitrary selected teams

Acceptance criteria:

* one feature row can be generated for every training match
* one feature row can be generated for any valid UI matchup
* no target leakage exists

## Milestone 3: Baseline Score Model

Deliverables:

* simple Poisson model
* scoreline matrix generation
* top scoreline extraction

Acceptance criteria:

* expected goals are positive
* scoreline probabilities sum to 1
* prediction response can be generated for a selected matchup

## Milestone 4: Dixon-Coles Model

Deliverables:

* Dixon-Coles adjustment
* validation comparison against simple Poisson
* selected model decision

Acceptance criteria:

* model trains without numerical instability
* selected model is at least as reliable as the baseline
* fallback behavior is documented

## Milestone 5: Prediction Service

Deliverables:

* `predict_match` function or API endpoint
* response formatter
* confidence label logic
* validation errors

Acceptance criteria:

* valid matchups return main score, probability, confidence, alternatives, and outcome probabilities
* invalid matchups return user-friendly errors

## Milestone 6: Simple UI

Deliverables:

* team selectors
* venue selector
* predict button
* result display
* error/loading states

Acceptance criteria:

* non-technical users can generate a prediction without reading documentation
* result display is understandable at a glance
* UI does not expose unnecessary model internals

---

## 18. Recommended Implementation Order

Build in this order:

```text
1. Load and validate data
2. Normalize team names
3. Build completed-match training dataset
4. Build internal Elo
5. Build rolling form features
6. Build attack/defense features
7. Train simple Poisson baseline
8. Generate scoreline probability matrix
9. Format prediction response
10. Add confidence labels and alternative scorelines
11. Add simple UI
12. Backtest on historical World Cups
13. Add Dixon-Coles adjustment if it improves validation
14. Save final model artifact
15. Polish UI states and wording
```

---

## 19. Risks and Limitations

## 19.1 Exact Score Prediction Is Hard

Risk:

```text
Exact football scores are low-probability events, even for the best prediction.
```

Mitigation:

```text
Show probability, confidence label, and alternative scorelines instead of presenting one score as certain.
```

## 19.2 No External Team Strength Data

Risk:

```text
The model may underestimate rapidly improving or declining teams.
```

Mitigation:

```text
Use time decay and recent-form features.
```

## 19.3 No Player Availability

Risk:

```text
Predictions may be inaccurate for teams with major squad changes, injuries, suspensions, or rotated lineups.
```

Mitigation:

```text
Keep predictions as ballpark probabilities, not betting-grade forecasts.
```

## 19.4 International Football Sample Size

Risk:

```text
National teams play fewer matches than clubs, so team parameters can overfit.
```

Mitigation:

```text
Use regularization, time decay, and validation on historical World Cups.
```

## 19.5 UI Misinterpretation

Risk:

```text
Users may read the top scoreline as a guarantee.
```

Mitigation:

```text
Always display exact-score probability, confidence label, and alternative scorelines near the main prediction.
```

---

## 20. Final Version 1 Definition

Version 1 is complete when a user can:

1. Open the app.
2. Select two World Cup 2026 teams.
3. Click `Predict`.
4. See the most likely score in `home goals - away goals` format.
5. See how likely that exact score is.
6. See a simple confidence hint.
7. See 3 to 5 other plausible scorelines.

The recommended first production model is:

```text
simple Poisson or Dixon-Coles adjusted Poisson, whichever validates better
```

with these core features:

```text
internal Elo
rolling recent form
rolling goals for / goals against
attack and defense strength
venue / neutral / host-country flags
optional basic scorer-profile features
```

The product should optimize for clarity, trust, and ease of use rather than broad tournament simulation features.

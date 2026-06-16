from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wc_predictor.data import DataValidationError
from wc_predictor.streamlit_support import (
    StreamlitAppResources,
    build_streamlit_resources,
    format_percent,
)


VENUE_OPTIONS = ["Neutral", "Team 1 host advantage", "Team 2 host advantage"]


@st.cache_resource(show_spinner=False)
def get_resources() -> StreamlitAppResources:
    return build_streamlit_resources(PROJECT_ROOT)


def main() -> None:
    st.set_page_config(
        page_title="World Cup 2026 Score Prediction",
        layout="centered",
    )

    st.title("World Cup 2026 Score Prediction")
    st.caption("Select two teams and get a simple score prediction.")

    try:
        with st.spinner("Loading prediction model..."):
            resources = get_resources()
    except Exception as exc:
        st.error("The prediction app could not start.")
        st.info(
            "Train the model artifact first with "
            "`python -m wc_predictor.models.train_app_model`, then restart Streamlit."
        )
        with st.expander("Error details"):
            st.code(str(exc))
        return

    if not resources.teams:
        st.error("No teams are configured yet.")
        return

    team_1, team_2, venue_mode = render_controls(resources.teams)
    can_predict = team_1 != team_2

    if not can_predict:
        st.warning("Select two different teams.")

    predict_clicked = st.button(
        "Predict",
        type="primary",
        disabled=not can_predict,
        use_container_width=True,
    )

    if not predict_clicked:
        st.info("Choose two teams, then click Predict.")
        return

    try:
        with st.spinner("Calculating prediction..."):
            prediction = resources.predictor.predict_match(
                home_team=team_1,
                away_team=team_2,
                venue_mode=venue_mode,
                prediction_date=resources.feature_cutoff_date,
            )
    except DataValidationError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error("Prediction could not be generated.")
        with st.expander("Error details"):
            st.code(str(exc))
        return

    render_prediction(prediction)
    render_model_note(resources)


def render_controls(teams: list[str]) -> tuple[str, str, str]:
    left, middle = st.columns(2)
    with left:
        team_1 = st.selectbox("Team 1", teams, index=0)
    with middle:
        default_index = 1 if len(teams) > 1 else 0
        team_2 = st.selectbox("Team 2", teams, index=default_index)

    venue_mode = st.selectbox("Venue", VENUE_OPTIONS, index=0)
    return team_1, team_2, venue_mode


def render_prediction(prediction: dict[str, object]) -> None:
    st.divider()
    st.subheader("Most likely score")
    st.markdown(f"## {prediction['most_likely_score']}")

    probability_col, confidence_col = st.columns(2)
    with probability_col:
        st.metric(
            "Chance of this exact score",
            format_percent(float(prediction["most_likely_score_prob"])),
        )
    with confidence_col:
        st.metric("Confidence", str(prediction["confidence_label"]))

    st.caption(str(prediction["confidence_explanation"]))

    low_history_flags = prediction.get("low_history_flags", {})
    if isinstance(low_history_flags, dict) and any(low_history_flags.values()):
        st.warning("One of these teams has limited recent match history in the model.")

    render_scoreline_table(prediction)
    render_outcome_chances(prediction)
    render_expected_goals(prediction)
    st.caption("Predictions are probabilities, not guarantees.")


def render_scoreline_table(prediction: dict[str, object]) -> None:
    st.subheader("Other possible scores")
    scorelines = prediction["top_scorelines"]
    rows = [
        {
            "Rank": item["rank"],
            "Score": f"{prediction['home_team']} {item['home_goals']} - {item['away_goals']} {prediction['away_team']}",
            "Chance": format_percent(float(item["probability"])),
        }
        for item in scorelines
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_outcome_chances(prediction: dict[str, object]) -> None:
    st.subheader("Match outcome chances")
    outcomes = [
        (f"{prediction['home_team']} win", float(prediction["prob_home_win"])),
        ("Draw", float(prediction["prob_draw"])),
        (f"{prediction['away_team']} win", float(prediction["prob_away_win"])),
    ]

    for label, probability in outcomes:
        st.write(f"{label}: {format_percent(probability)}")
        st.progress(probability)


def render_expected_goals(prediction: dict[str, object]) -> None:
    left, right = st.columns(2)
    with left:
        st.metric(
            f"{prediction['home_team']} expected goals",
            f"{float(prediction['expected_home_goals']):.2f}",
        )
    with right:
        st.metric(
            f"{prediction['away_team']} expected goals",
            f"{float(prediction['expected_away_goals']):.2f}",
        )


def render_model_note(resources: StreamlitAppResources) -> None:
    with st.expander("About the model"):
        st.write(
            "This POC loads a pre-trained score model built from historical international "
            "results and converts expected goals into scoreline probabilities."
        )
        st.write(f"Training matches used: {resources.training_match_count}")
        st.write(f"Feature cutoff date: {resources.feature_cutoff_date}")


if __name__ == "__main__":
    main()

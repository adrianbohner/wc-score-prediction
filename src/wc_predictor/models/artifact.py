from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from wc_predictor.data import DataValidationError


DEFAULT_MODEL_ARTIFACT_PATH = Path("models") / "match_score_model.pkl"
ARTIFACT_VERSION = "3"


def save_model_artifact(resources: Any, path: str | Path) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as file:
        pickle.dump(resources, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_artifact(path: str | Path) -> Any:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise DataValidationError(
            f"model artifact was not found at {artifact_path}. "
            "Run `python -m wc_predictor.models.train_app_model` before starting Streamlit."
        )

    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)

    required_attributes = {"teams", "predictor", "training_match_count", "feature_cutoff_date"}
    if not all(hasattr(artifact, attribute) for attribute in required_attributes):
        raise DataValidationError(f"invalid model artifact format: {artifact_path}")
    return artifact

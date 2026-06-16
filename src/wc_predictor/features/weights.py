from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sample_weights(
    training_features: pd.DataFrame,
    reference_date: pd.Timestamp,
    half_life_days: float = 1460.0,
    friendly_weight: float = 0.5,
) -> np.ndarray:
    """Exponential time-decay * tournament importance weights.

    Weights are normalised to mean 1.0 so the alpha scale is preserved.
    """
    dates = pd.to_datetime(training_features["date"])
    delta_days = (reference_date - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
    decay = np.exp(-np.log(2) * delta_days / max(half_life_days, 1.0))

    tournament_multiplier = np.where(
        training_features["tournament"].str.contains("Friendly", na=False),
        friendly_weight,
        1.0,
    )

    weights = decay * tournament_multiplier
    mean_w = float(weights.mean())
    if mean_w > 0:
        weights = weights / mean_w
    return weights.astype(float)

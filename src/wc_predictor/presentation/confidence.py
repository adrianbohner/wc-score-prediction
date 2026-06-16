from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceThresholds:
    low_max: float = 0.12
    medium_max: float = 0.18


CONFIDENCE_EXPLANATIONS = {
    "Low": "Several scorelines are similarly likely. Treat this as a weak favorite.",
    "Medium": "This is the clearest scoreline, but nearby scores remain plausible.",
    "High": "The model sees this as a relatively strong exact-score pick.",
}


def confidence_label(
    probability: float,
    thresholds: ConfidenceThresholds | None = None,
) -> str:
    thresholds = thresholds or ConfidenceThresholds()
    if probability < thresholds.low_max:
        return "Low"
    if probability <= thresholds.medium_max:
        return "Medium"
    return "High"


def confidence_explanation(label: str) -> str:
    return CONFIDENCE_EXPLANATIONS.get(label, CONFIDENCE_EXPLANATIONS["Low"])


def confidence_hint(
    probability: float,
    thresholds: ConfidenceThresholds | None = None,
) -> dict[str, str]:
    label = confidence_label(probability, thresholds)
    return {
        "confidence_label": label,
        "confidence_explanation": confidence_explanation(label),
    }


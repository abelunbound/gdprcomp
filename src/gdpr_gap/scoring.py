"""Turn criterion verdicts into a 0–100 score, a band, and a 220-point score.

Yes, Somewhat, and No contribute the values in ScoringConfig. The percentage
score is 100 times the weighted sum divided by the total weight. The 220-point
score is that same fraction of 220, so 22 equally weighted Yes verdicts score
220, matching the archived evaluator. A score of 80 or more is high, 50–79 is
mid, and below 50 is low.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from gdpr_gap.config import ScoringConfig

POINTS_SCALE = 220
VERDICTS = ("Yes", "Somewhat", "No")
Band = Literal["high", "mid", "low"]


class ScoringError(Exception):
    """Raised when verdicts or weights cannot be scored."""


class ScoredCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    verdict: Literal["Yes", "Somewhat", "No"]
    weight: float
    value: float
    weighted_value: float


class PolicyScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    points: float
    band: Band
    weighted_sum: float
    weight_total: float
    criteria: list[ScoredCriterion]


def score_policy(
    verdicts: Mapping[str, str],
    weights: Mapping[str, float],
    scoring: ScoringConfig,
) -> PolicyScore:
    """Score one policy. Every verdict needs a positive weight, and the reverse."""
    if not verdicts:
        raise ScoringError("at least one criterion verdict is required")
    if set(verdicts) != set(weights):
        raise ScoringError("verdicts and weights must name the same criteria")

    scored: list[ScoredCriterion] = []
    for criterion_id, verdict in verdicts.items():
        weight = weights[criterion_id]
        if not math.isfinite(weight) or weight <= 0:
            raise ScoringError(
                f"weight for {criterion_id} must be positive and finite, got {weight}"
            )
        known = _known_verdict(criterion_id, verdict)
        value = _verdict_value(known, scoring)
        scored.append(
            ScoredCriterion(
                criterion_id=criterion_id,
                verdict=known,
                weight=weight,
                value=value,
                weighted_value=value * weight,
            )
        )

    weight_total = sum(item.weight for item in scored)
    if weight_total <= 0:
        raise ScoringError(f"total weight must be positive, got {weight_total}")
    weighted_sum = sum(item.weighted_value for item in scored)
    fraction = weighted_sum / weight_total
    return PolicyScore(
        score=100 * fraction,
        points=POINTS_SCALE * fraction,
        band=band_for(100 * fraction, scoring),
        weighted_sum=weighted_sum,
        weight_total=weight_total,
        criteria=scored,
    )


def band_for(score: float, scoring: ScoringConfig) -> Band:
    """Map a 0–100 score onto the configured bands."""
    if score >= scoring.high_band_min:
        return "high"
    if score >= scoring.mid_band_min:
        return "mid"
    return "low"


def _known_verdict(criterion_id: str, verdict: str) -> Literal["Yes", "Somewhat", "No"]:
    if verdict == "Yes":
        return "Yes"
    if verdict == "Somewhat":
        return "Somewhat"
    if verdict == "No":
        return "No"
    allowed = ", ".join(VERDICTS)
    raise ScoringError(
        f"unknown verdict for {criterion_id}: {verdict!r}. Expected one of {allowed}."
    )


def _verdict_value(verdict: Literal["Yes", "Somewhat", "No"], scoring: ScoringConfig) -> float:
    if verdict == "Yes":
        return scoring.yes_score
    if verdict == "Somewhat":
        return scoring.somewhat_score
    return scoring.no_score

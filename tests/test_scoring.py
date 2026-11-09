import math

import pytest

from gdpr_gap.config import ScoringConfig
from gdpr_gap.scoring import POINTS_SCALE, ScoringError, band_for, score_policy


def _scoring() -> ScoringConfig:
    return ScoringConfig(
        yes_score=1,
        somewhat_score=0.5,
        no_score=0,
        high_band_min=80,
        mid_band_min=50,
    )


def test_all_yes_scores_100_and_220_points() -> None:
    verdicts = {f"c{index}": "Yes" for index in range(22)}
    weights = {criterion_id: 1.0 for criterion_id in verdicts}

    result = score_policy(verdicts, weights, _scoring())

    assert result.score == 100
    assert result.points == POINTS_SCALE
    assert result.band == "high"
    assert result.weight_total == 22
    assert result.weighted_sum == 22


def test_all_no_scores_zero() -> None:
    verdicts = {f"c{index}": "No" for index in range(22)}
    weights = {criterion_id: 1.0 for criterion_id in verdicts}

    result = score_policy(verdicts, weights, _scoring())

    assert result.score == 0
    assert result.points == 0
    assert result.band == "low"
    assert result.weighted_sum == 0


def test_mixed_weights_use_the_weighted_fraction() -> None:
    verdicts = {"records": "Yes", "consent": "Somewhat", "security": "No"}
    weights = {"records": 2.0, "consent": 1.0, "security": 3.0}

    result = score_policy(verdicts, weights, _scoring())

    assert result.weighted_sum == pytest.approx(2.5)
    assert result.weight_total == 6
    assert result.score == pytest.approx(100 * 2.5 / 6)
    assert result.points == pytest.approx(POINTS_SCALE * 2.5 / 6)
    assert result.band == "low"
    assert [item.criterion_id for item in result.criteria] == [
        "records",
        "consent",
        "security",
    ]
    assert result.criteria[1].weighted_value == pytest.approx(0.5)


def test_all_somewhat_converts_to_half_of_220() -> None:
    verdicts = {f"c{index}": "Somewhat" for index in range(22)}
    weights = {criterion_id: 1.0 for criterion_id in verdicts}

    result = score_policy(verdicts, weights, _scoring())

    assert result.score == 50
    assert result.points == POINTS_SCALE / 2
    assert result.band == "mid"


def test_band_boundaries() -> None:
    scoring = _scoring()

    exactly_high = score_policy({"a": "Yes", "b": "No"}, {"a": 4.0, "b": 1.0}, scoring)
    just_below_high = score_policy({"a": "Yes", "b": "No"}, {"a": 799.0, "b": 201.0}, scoring)
    exactly_mid = score_policy({"a": "Yes", "b": "No"}, {"a": 1.0, "b": 1.0}, scoring)
    top_of_mid = score_policy({"a": "Yes", "b": "No"}, {"a": 79.0, "b": 21.0}, scoring)
    just_below_mid = score_policy({"a": "Yes", "b": "No"}, {"a": 49.0, "b": 51.0}, scoring)

    assert exactly_high.score == 80
    assert exactly_high.band == "high"
    assert just_below_high.score == pytest.approx(79.9)
    assert just_below_high.band == "mid"
    assert exactly_mid.score == 50
    assert exactly_mid.band == "mid"
    assert top_of_mid.score == 79
    assert top_of_mid.band == "mid"
    assert just_below_mid.score == 49
    assert just_below_mid.band == "low"
    assert band_for(80, scoring) == "high"
    assert band_for(50, scoring) == "mid"
    assert band_for(49.999, scoring) == "low"


def test_zero_negative_and_unknown_inputs_are_rejected() -> None:
    scoring = _scoring()

    with pytest.raises(ScoringError, match="at least one"):
        score_policy({}, {}, scoring)
    with pytest.raises(ScoringError, match="positive"):
        score_policy({"a": "Yes"}, {"a": 0}, scoring)
    with pytest.raises(ScoringError, match="positive"):
        score_policy({"a": "Yes"}, {"a": -1}, scoring)
    with pytest.raises(ScoringError, match="finite"):
        score_policy({"a": "Yes"}, {"a": math.inf}, scoring)
    with pytest.raises(ScoringError, match="unknown verdict"):
        score_policy({"a": "yes"}, {"a": 1}, scoring)
    with pytest.raises(ScoringError, match="same criteria"):
        score_policy({"a": "Yes"}, {}, scoring)
    with pytest.raises(ScoringError, match="same criteria"):
        score_policy({"a": "Yes"}, {"a": 1, "b": 1}, scoring)

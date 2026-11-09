import pytest

from gdpr_gap.config import ScoringConfig
from gdpr_gap.gaps import gap_ids, is_gap
from gdpr_gap.scoring import score_policy


def test_only_an_exact_yes_is_not_a_gap() -> None:
    assert is_gap("Yes") is False


@pytest.mark.parametrize("verdict", ["Somewhat", "No", "yes", "YES", "Yes ", ""])
def test_every_other_verdict_is_a_gap(verdict: str) -> None:
    assert is_gap(verdict) is True


def test_gap_ids_keep_mapping_order_and_skip_yes() -> None:
    verdicts = {"records": "Yes", "consent": "Somewhat", "security": "No", "access": "Yes"}

    assert gap_ids(verdicts) == ["consent", "security"]


def test_scored_policy_reports_gaps_through_the_helper() -> None:
    scoring = ScoringConfig(
        yes_score=1,
        somewhat_score=0.5,
        no_score=0,
        high_band_min=80,
        mid_band_min=50,
    )
    verdicts = {"records": "Yes", "consent": "Somewhat", "security": "No"}
    weights = {"records": 1.0, "consent": 1.0, "security": 1.0}

    result = score_policy(verdicts, weights, scoring)

    assert result.gap_ids() == gap_ids(verdicts)
    assert [is_gap(item.verdict) for item in result.criteria] == [False, True, True]

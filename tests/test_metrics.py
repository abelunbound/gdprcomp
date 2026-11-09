import pytest

from gdpr_gap.metrics import MetricsError, Verdict, VerdictPair, classification_metrics

JUDGE = "claude-haiku-5-5"
GOLD = "claude-sonnet-5-5"


def _pair(policy_id: str, criterion_id: str, verdict: Verdict) -> VerdictPair:
    return VerdictPair(policy_id=policy_id, criterion_id=criterion_id, verdict=verdict)


def test_metrics_score_each_criterion_and_the_overall_set() -> None:
    predictions = [
        _pair("745", "purpose_limitation", "Yes"),
        _pair("745", "data_processing_records", "Somewhat"),
    ]
    labels = [
        _pair("745", "purpose_limitation", "Yes"),
        _pair("745", "data_processing_records", "No"),
    ]
    report = classification_metrics(
        predictions,
        labels,
        prediction_model_id=JUDGE,
        label_model_id=GOLD,
    )
    by_id = {item.criterion_id: item for item in report.criteria}
    agreed = by_id["purpose_limitation"]
    shifted = by_id["data_processing_records"]

    assert report.label_source == "model-produced"
    assert report.prediction_model_id == JUDGE
    assert agreed.gap.f1 == 0
    assert agreed.weighted_kappa == 1
    assert agreed.confusion.counts[2][2] == 1
    assert shifted.gap.true_positive == 1
    assert shifted.gap.f1 == 1
    assert report.overall.pair_count == 2
    assert report.overall.gap.true_positive == 1
    assert report.overall.gap.false_negative == 0
    assert report.overall.gap.precision == 1
    assert report.overall.gap.recall == 1
    assert report.overall.gap.f1 == 1
    assert report.overall.weighted_kappa == pytest.approx(0.5)
    assert report.overall.confusion.labels == ["No", "Somewhat", "Yes"]
    assert report.overall.confusion.counts[0][1] == 1
    assert report.overall.confusion.counts[2][2] == 1


def test_same_system_is_rejected_before_scoring() -> None:
    rows = [_pair("745", "purpose_limitation", "Yes")]
    with pytest.raises(MetricsError, match="same system"):
        classification_metrics(
            rows,
            rows,
            prediction_model_id=JUDGE,
            label_model_id=JUDGE,
        )
    with pytest.raises(MetricsError, match="same policies"):
        classification_metrics(
            rows,
            [_pair("747", "purpose_limitation", "Yes")],
            prediction_model_id=JUDGE,
            label_model_id=GOLD,
        )

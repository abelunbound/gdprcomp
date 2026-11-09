import pytest

from gdpr_gap.metrics import (
    MetricsError,
    Verdict,
    VerdictPair,
    bootstrap_intervals,
    classification_metrics,
    expand_policy_draw,
    format_bootstrap,
    policy_draws,
)

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


def test_bootstrap_resamples_policies_and_reports_the_intervals() -> None:
    predictions, labels = _twenty_policies()
    report = bootstrap_intervals(
        predictions,
        labels,
        prediction_model_id=JUDGE,
        label_model_id=GOLD,
        seed=7,
        resamples=200,
    )
    again = bootstrap_intervals(
        predictions,
        labels,
        prediction_model_id=JUDGE,
        label_model_id=GOLD,
        seed=7,
        resamples=200,
    )
    scored = classification_metrics(
        predictions,
        labels,
        prediction_model_id=JUDGE,
        label_model_id=GOLD,
    )
    text = format_bootstrap(report)

    assert again.model_dump() == report.model_dump()
    assert report.policy_count == 20
    assert report.label_source == "model-produced"
    assert report.gap_f1.estimate == scored.overall.gap.f1
    assert report.macro_f1.estimate == scored.overall.macro_f1
    assert report.weighted_kappa.estimate == scored.overall.weighted_kappa
    assert "20 policies" in text
    assert "Gap F1 95% interval:" in text
    assert policy_draws([f"{index:02d}" for index in range(20)], seed=7, resamples=200) != (
        policy_draws([f"{index:02d}" for index in range(20)], seed=8, resamples=200)
    )


def test_a_repeated_policy_keeps_both_of_its_judgements() -> None:
    rows: dict[str, list[tuple[Verdict, Verdict]]] = {
        "745": [("Yes", "Yes"), ("No", "Somewhat")],
        "747": [("No", "No")],
    }
    expanded = expand_policy_draw(("745", "745"), rows)
    assert expanded == [
        ("Yes", "Yes"),
        ("No", "Somewhat"),
        ("Yes", "Yes"),
        ("No", "Somewhat"),
    ]


def test_bootstrap_rejects_one_policy_and_a_bad_seed() -> None:
    rows = [_pair("745", "purpose_limitation", "Yes")]
    with pytest.raises(MetricsError, match="at least two policies"):
        bootstrap_intervals(
            rows,
            rows,
            prediction_model_id=JUDGE,
            label_model_id=GOLD,
            seed=0,
        )
    with pytest.raises(MetricsError, match="same system"):
        bootstrap_intervals(
            rows,
            rows,
            prediction_model_id=JUDGE,
            label_model_id=JUDGE,
            seed=0,
        )
    with pytest.raises(MetricsError, match="zero or greater"):
        policy_draws(["745", "747"], seed=-1, resamples=1)


def _twenty_policies() -> tuple[list[VerdictPair], list[VerdictPair]]:
    predictions: list[VerdictPair] = []
    labels: list[VerdictPair] = []
    for index in range(20):
        policy_id = f"{index:02d}"
        predicted: Verdict = "Yes" if index % 2 == 0 else "Somewhat"
        gold: Verdict = "Yes" if index % 3 == 0 else "No"
        predictions.append(_pair(policy_id, "purpose_limitation", predicted))
        labels.append(_pair(policy_id, "purpose_limitation", gold))
        predictions.append(_pair(policy_id, "data_processing_records", "No"))
        labels.append(_pair(policy_id, "data_processing_records", "No"))
    return predictions, labels

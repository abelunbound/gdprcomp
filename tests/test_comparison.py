import pytest

from gdpr_gap.comparison import (
    ComparisonError,
    MethodResult,
    compare_methods,
    export_comparison,
    format_comparison,
)
from gdpr_gap.metrics import Verdict, VerdictPair

JUDGE = "claude-haiku-5-5"
GOLD = "claude-sonnet-5-5"


def _pair(policy_id: str, verdict: Verdict) -> VerdictPair:
    return VerdictPair(policy_id=policy_id, criterion_id="purpose_limitation", verdict=verdict)


def _labels() -> list[VerdictPair]:
    return [_pair("745", "Yes"), _pair("747", "No")]


def test_each_metric_names_the_better_method() -> None:
    report = compare_methods(
        [
            MethodResult(
                method="whole_policy_llm",
                predictions=_labels(),
                runtime_seconds=10,
                cost_usd=1,
                repeat_agreement=0.5,
                prediction_model_id=JUDGE,
            ),
            MethodResult(
                method="hybrid",
                predictions=[_pair("745", "No"), _pair("747", "Yes")],
                runtime_seconds=1,
                cost_usd=0.5,
                repeat_agreement=1,
                prediction_model_id=JUDGE,
            ),
            MethodResult(
                method="legal_bert_only",
                runtime_seconds=0.1,
                repeat_agreement=0.2,
            ),
        ],
        _labels(),
        label_model_id=GOLD,
    )
    by_metric = {item.metric: item for item in report.winners}
    text = format_comparison(report)

    assert report.label_source == "model-produced"
    assert by_metric["accuracy"].winner == "whole_policy_llm"
    assert by_metric["runtime"].winner == "legal_bert_only"
    assert by_metric["cost"].winner == "hybrid"
    assert by_metric["repeat_consistency"].winner == "hybrid"
    assert "Accuracy: Whole-policy LLM (1.000)" in text
    assert "Runtime: Legal-BERT only (0.100 seconds)" in text
    assert "Alternative embeddings: not available" in text
    assert "Legal-BERT only: retrieval signal, no verdict" in text
    assert "Hybrid: not run" not in text


def test_a_method_that_was_not_run_cannot_win() -> None:
    report = compare_methods(
        [
            MethodResult(
                method="whole_policy_llm",
                predictions=_labels(),
                runtime_seconds=10,
                cost_usd=1,
                repeat_agreement=0.5,
                prediction_model_id=JUDGE,
            ),
            MethodResult(method="legal_bert_only", runtime_seconds=0.1),
        ],
        _labels(),
        label_model_id=GOLD,
    )
    text = format_comparison(report)

    assert all(item.winner != "hybrid" for item in report.winners)
    assert "Hybrid: not run" in text
    assert "Cost: Whole-policy LLM (1.000 USD)" in text
    assert "Repeat consistency: Whole-policy LLM (0.500)" in text


def test_a_tie_has_no_single_winner() -> None:
    shared = MethodResult(
        method="whole_policy_llm",
        predictions=_labels(),
        runtime_seconds=1,
        prediction_model_id=JUDGE,
    )
    report = compare_methods(
        [
            shared,
            shared.model_copy(
                update={
                    "method": "hybrid",
                    "predictions": [_pair("745", "Yes"), _pair("747", "No")],
                }
            ),
        ],
        _labels(),
        label_model_id=GOLD,
    )
    accuracy = next(item for item in report.winners if item.metric == "accuracy")

    assert accuracy.winner is None
    assert accuracy.tied == ["hybrid", "whole_policy_llm"]
    assert "Accuracy: tie between Hybrid and Whole-policy LLM (1.000)" in format_comparison(report)


def test_same_system_and_missing_coverage_are_rejected() -> None:
    scored = MethodResult(
        method="whole_policy_llm",
        predictions=_labels(),
        runtime_seconds=1,
        prediction_model_id=GOLD,
    )
    with pytest.raises(ComparisonError, match="same system"):
        compare_methods([scored], _labels(), label_model_id=GOLD)
    with pytest.raises(ComparisonError, match="same policies"):
        compare_methods(
            [
                scored.model_copy(
                    update={
                        "prediction_model_id": JUDGE,
                        "predictions": [_pair("745", "Yes")],
                    }
                )
            ],
            _labels(),
            label_model_id=GOLD,
        )
    with pytest.raises(ComparisonError, match="no verdict"):
        compare_methods(
            [MethodResult(method="legal_bert_only", predictions=_labels(), runtime_seconds=1)],
            _labels(),
            label_model_id=GOLD,
        )


def test_export_keeps_the_winner_and_the_unavailable_method() -> None:
    report = compare_methods(
        [
            MethodResult(
                method="whole_policy_llm",
                predictions=_labels(),
                runtime_seconds=2,
                prediction_model_id=JUDGE,
            )
        ],
        _labels(),
        label_model_id=GOLD,
    )
    exported = export_comparison(report)

    assert '"winner": "whole_policy_llm"' in exported
    assert '"method": "alternative_embeddings"' in exported
    assert '"status": "unavailable"' in exported

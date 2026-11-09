import json
from datetime import UTC, datetime

import pytest

from gdpr_gap.annotation import (
    Adjudication,
    Annotation,
    AnnotationError,
    Verdict,
    add_annotation,
    adjudicate,
    adopt_agreement,
    empty_set,
    export_annotations,
    export_blind,
    final_labels,
    gold_labels,
    import_annotations,
    merge_annotations,
)

WHEN = datetime(2026, 11, 9, 21, 0, tzinfo=UTC)
JUDGE = "claude-haiku-5-5"
RATER_A = "claude-sonnet-5-5"
RATER_B = "second-labeler"
ADJUDICATOR = "adjudicator"


def _label(
    annotator_id: str,
    verdict: Verdict,
    notes: str,
    criterion_id: str = "data_processing_records",
) -> Annotation:
    return Annotation(
        policy_id="745",
        criterion_id=criterion_id,
        annotator_id=annotator_id,
        verdict=verdict,
        notes=notes,
        timestamp=WHEN,
    )


def test_annotators_label_blind_and_adjudication_is_the_final_verdict() -> None:
    first = add_annotation(empty_set(JUDGE), _label(RATER_A, "Yes", "notes-from-a"))
    second = add_annotation(empty_set(JUDGE), _label(RATER_B, "No", "notes-from-b"))
    combined = merge_annotations([first, second])
    blinded = export_blind(combined, RATER_A)
    final = adjudicate(
        combined,
        adjudicator_id=ADJUDICATOR,
        policy_id="745",
        criterion_id="data_processing_records",
        verdict="Somewhat",
        notes="Split the difference.",
        timestamp=WHEN,
    )
    labels = final_labels(final)

    assert "notes-from-b" not in blinded
    assert RATER_B not in blinded
    assert "adjudicated_verdict" not in blinded
    assert import_annotations(blinded).annotations[0].annotator_id == RATER_A
    assert len(labels) == 1
    assert labels[0].adjudicated_verdict == "Somewhat"
    assert labels[0].annotator_ids == [RATER_A, RATER_B]
    assert labels[0].label_source == "model-produced"
    assert export_blind(final, RATER_A) == blinded
    restored = import_annotations(export_annotations(final))
    assert restored.model_dump(mode="json") == final.model_dump(mode="json")


def test_agreement_is_adopted_and_invalid_labels_are_rejected() -> None:
    agreed = merge_annotations(
        [
            add_annotation(
                empty_set(JUDGE),
                _label(RATER_A, "Yes", "same-a", "purpose_limitation"),
            ),
            add_annotation(
                empty_set(JUDGE),
                _label(RATER_B, "Yes", "same-b", "purpose_limitation"),
            ),
        ]
    )
    adopted = adopt_agreement(agreed, adjudicator_id=ADJUDICATOR, timestamp=WHEN)
    assert final_labels(adopted)[0].adjudicated_verdict == "Yes"

    with pytest.raises(AnnotationError, match="judge model"):
        add_annotation(empty_set(JUDGE), _label(JUDGE, "Yes", "judge"))
    with pytest.raises(AnnotationError, match="already labelled"):
        add_annotation(agreed, _label(RATER_A, "No", "again", "purpose_limitation"))
    with pytest.raises(AnnotationError, match="Two annotators"):
        add_annotation(agreed, _label("third-rater", "No", "extra", "purpose_limitation"))
    with pytest.raises(AnnotationError, match="must not be one of the two"):
        adjudicate(
            agreed,
            adjudicator_id=RATER_A,
            policy_id="745",
            criterion_id="purpose_limitation",
            verdict="Yes",
            notes="",
            timestamp=WHEN,
        )
    with pytest.raises(AnnotationError, match="two independent"):
        adjudicate(
            add_annotation(empty_set(JUDGE), _label(RATER_A, "Yes", "only")),
            adjudicator_id=ADJUDICATOR,
            policy_id="745",
            criterion_id="data_processing_records",
            verdict="Yes",
            notes="",
            timestamp=WHEN,
        )
    with pytest.raises(AnnotationError, match="schema"):
        payload = json.loads(export_annotations(agreed))
        payload["label_source"] = "expert"
        import_annotations(json.dumps(payload))


def test_sonnet_alone_is_the_gold_label() -> None:
    gold = add_annotation(
        empty_set(JUDGE, gold_model_id=RATER_A),
        _label(RATER_A, "Somewhat", "sonnet-only"),
    )
    labels = gold_labels(gold)

    assert labels[0].annotator_id == RATER_A
    assert labels[0].verdict == "Somewhat"
    assert labels[0].label_source == "model-produced"
    assert final_labels(gold) == []
    restored = import_annotations(export_annotations(gold))
    assert gold_labels(restored)[0].verdict == "Somewhat"

    with pytest.raises(AnnotationError, match="Only the gold labeller"):
        add_annotation(gold, _label(RATER_B, "No", "second"))
    with pytest.raises(AnnotationError, match="gold labeller must differ"):
        empty_set(JUDGE, gold_model_id=JUDGE)


def test_adjudication_record_keeps_the_two_annotators() -> None:
    record = Adjudication(
        policy_id="745",
        criterion_id="data_processing_records",
        adjudicator_id=ADJUDICATOR,
        adjudicated_verdict="No",
        notes="",
        timestamp=WHEN,
        annotator_ids=[RATER_B, RATER_A],
    )
    assert record.annotator_ids == [RATER_A, RATER_B]

import json
from pathlib import Path
from typing import Literal

import pytest

from gdpr_gap.exports import ExportError, export_classification, export_feedback_quality
from gdpr_gap.feedback import CitationReport, CriterionFeedback
from gdpr_gap.judge import PolicyJudgement
from gdpr_gap.llm import CriterionJudgement, ModelResult, TokenUse
from gdpr_gap.scoring import PolicyScore, ScoredCriterion

POLICY_ID = "example.co.uk"
FEEDBACK_KEYS = {
    "policy_id",
    "criterion_id",
    "suggestion",
    "evidence_quotes",
    "citation_support",
    "unsupported_quotes",
    "rater_correctness",
    "rater_actionability",
}


def _feedback() -> list[CriterionFeedback]:
    return [
        CriterionFeedback(
            criterion_id="consent_mechanisms",
            suggestion="State how consent is withdrawn.",
            evidence_quotes=["We ask for consent."],
        )
    ]


def _citations(
    status: Literal["supported", "unsupported"] = "supported",
) -> list[CitationReport]:
    unsupported = [] if status == "supported" else ["We sell personal data."]
    return [
        CitationReport(
            criterion_id="consent_mechanisms",
            citation_support=status,
            unsupported_quotes=unsupported,
        )
    ]


def _judgement() -> PolicyJudgement:
    verdicts = []
    for index in range(22):
        criterion_id = f"criterion_{index}"
        verdicts.append(
            ModelResult(
                judgement=CriterionJudgement(
                    criterion_id=criterion_id,
                    verdict="Yes",
                    justification="The policy covers this criterion.",
                    evidence_quotes=["We ask for consent."],
                ),
                model_id="claude-haiku-5-5",
                usage=TokenUse(input_tokens=1, output_tokens=1),
                prompt_id="whole_policy_judge",
                prompt_version="v1",
            )
        )
    return PolicyJudgement(
        prompt_id="whole_policy_judge",
        prompt_version="v1",
        model_id="claude-haiku-5-5",
        verdicts=verdicts,
    )


def _score() -> PolicyScore:
    criteria = [
        ScoredCriterion(
            criterion_id=f"criterion_{index}",
            verdict="Yes",
            weight=1,
            value=1,
            weighted_value=1,
        )
        for index in range(22)
    ]
    return PolicyScore(
        score=100,
        points=220,
        band="high",
        weighted_sum=22,
        weight_total=22,
        criteria=criteria,
    )


def test_feedback_export_has_rater_and_citation_fields_and_no_f1(tmp_path: Path) -> None:
    document = export_feedback_quality(
        POLICY_ID,
        _feedback(),
        _citations("unsupported"),
        ratings={"consent_mechanisms": ("correct", "actionable")},
    )
    path = tmp_path / "feedback.json"
    path.write_text(document, encoding="utf-8")
    rows = json.loads(path.read_text(encoding="utf-8"))

    assert list(rows[0]) == sorted(FEEDBACK_KEYS)
    assert rows[0]["citation_support"] == "unsupported"
    assert rows[0]["unsupported_quotes"] == ["We sell personal data."]
    assert rows[0]["rater_correctness"] == "correct"
    assert rows[0]["rater_actionability"] == "actionable"
    assert "f1" not in rows[0]
    assert "verdict" not in rows[0]
    assert "score" not in rows[0]


def test_feedback_export_is_separate_from_classification(tmp_path: Path) -> None:
    feedback_path = tmp_path / "feedback.json"
    classification_path = tmp_path / "classification.json"
    feedback_path.write_text(
        export_feedback_quality(POLICY_ID, _feedback(), _citations()),
        encoding="utf-8",
    )
    classification_path.write_text(
        export_classification(POLICY_ID, _judgement(), _score()),
        encoding="utf-8",
    )

    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    classification = json.loads(classification_path.read_text(encoding="utf-8"))

    assert set(feedback[0]) == FEEDBACK_KEYS
    assert set(classification) == {"policy_id", "score", "points", "band", "verdicts"}
    assert "f1" not in classification
    assert "suggestion" not in classification
    assert "citation_support" not in classification
    assert "rater_correctness" not in classification
    assert feedback_path.read_bytes() != classification_path.read_bytes()


def test_unrated_feedback_leaves_rater_fields_empty() -> None:
    rows = json.loads(export_feedback_quality(POLICY_ID, _feedback(), _citations()))

    assert rows[0]["rater_correctness"] is None
    assert rows[0]["rater_actionability"] is None


def test_mismatched_feedback_and_citations_are_rejected() -> None:
    with pytest.raises(ExportError, match="same criteria"):
        export_feedback_quality(POLICY_ID, _feedback(), [])

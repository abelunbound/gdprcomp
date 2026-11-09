"""Separate exports for feedback quality and classification results.

The feedback file records rater correctness, rater actionability, and
citation-support status. It does not record verdicts, scores, or F1.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gdpr_gap.feedback import CitationReport, CriterionFeedback
from gdpr_gap.judge import PolicyJudgement
from gdpr_gap.scoring import PolicyScore

RaterCorrectness = Literal["correct", "incorrect"]
RaterActionability = Literal["actionable", "not_actionable"]


class ExportError(Exception):
    """Raised when an export cannot be built from the given records."""


class FeedbackQualityRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    suggestion: str = Field(min_length=1)
    evidence_quotes: list[str]
    citation_support: Literal["supported", "unsupported"]
    unsupported_quotes: list[str]
    rater_correctness: RaterCorrectness | None = None
    rater_actionability: RaterActionability | None = None


class ClassificationVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    verdict: Literal["Yes", "Somewhat", "No"]
    justification: str = Field(min_length=1)
    evidence_quotes: list[str]


class ClassificationExport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    score: float
    points: float
    band: Literal["high", "mid", "low"]
    verdicts: list[ClassificationVerdict]


def export_feedback_quality(
    policy_id: str,
    feedback: list[CriterionFeedback],
    citations: list[CitationReport],
    *,
    ratings: dict[str, tuple[RaterCorrectness | None, RaterActionability | None]] | None = None,
) -> str:
    """Return feedback-quality JSON. Classification fields are not included."""
    rows = feedback_quality_rows(policy_id, feedback, citations, ratings=ratings)
    payload = [row.model_dump() for row in rows]
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def export_classification(policy_id: str, judgement: PolicyJudgement, score: PolicyScore) -> str:
    """Return classification JSON. Feedback and F1 are not included."""
    document = ClassificationExport(
        policy_id=policy_id,
        score=score.score,
        points=score.points,
        band=score.band,
        verdicts=[
            ClassificationVerdict(
                criterion_id=item.judgement.criterion_id,
                verdict=item.judgement.verdict,
                justification=item.judgement.justification,
                evidence_quotes=item.judgement.evidence_quotes,
            )
            for item in judgement.verdicts
        ],
    )
    return document.model_dump_json(indent=2) + "\n"


def feedback_quality_rows(
    policy_id: str,
    feedback: list[CriterionFeedback],
    citations: list[CitationReport],
    *,
    ratings: dict[str, tuple[RaterCorrectness | None, RaterActionability | None]] | None = None,
) -> list[FeedbackQualityRow]:
    """Pair each suggestion with its citation status and optional rater fields."""
    by_id = {item.criterion_id: item for item in citations}
    if len(by_id) != len(citations):
        raise ExportError("Citation reports contain a duplicate criterion.")
    feedback_ids = [item.criterion_id for item in feedback]
    if len(set(feedback_ids)) != len(feedback_ids):
        raise ExportError("Feedback contains a duplicate criterion.")
    if set(by_id) != set(feedback_ids):
        raise ExportError("Feedback and citation reports must cover the same criteria.")
    given = ratings or {}
    rows: list[FeedbackQualityRow] = []
    for item in feedback:
        report = by_id[item.criterion_id]
        correctness, actionability = given.get(item.criterion_id, (None, None))
        rows.append(
            FeedbackQualityRow(
                policy_id=policy_id,
                criterion_id=item.criterion_id,
                suggestion=item.suggestion,
                evidence_quotes=item.evidence_quotes,
                citation_support=report.citation_support,
                unsupported_quotes=report.unsupported_quotes,
                rater_correctness=correctness,
                rater_actionability=actionability,
            )
        )
    return rows

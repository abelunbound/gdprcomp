"""Display model for the single-policy view.

Condition met is the verdict. Relevant text found is the evidence the model
cited. Those are separate. Feedback is shown only for gaps, using is_gap.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gdpr_gap.config import AppConfig
from gdpr_gap.criteria import load_criteria
from gdpr_gap.feedback import CitationReport, CriterionFeedback, PolicyReview, score_verdicts
from gdpr_gap.gaps import is_gap
from gdpr_gap.judge import PolicyJudgement
from gdpr_gap.llm import CriterionJudgement, ModelResult, TokenUse

METHOD: Literal["whole-policy LLM"] = "whole-policy LLM"
NOT_LEGAL_ADVICE: Literal["LLM output is not legal advice and is not ground truth."] = (
    "LLM output is not legal advice and is not ground truth."
)


class CriterionDisplay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1)
    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=1)
    condition_met: Literal["Yes", "Somewhat", "No"]
    relevant_text: list[str]
    feedback: str | None = None


class SinglePolicyDisplay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["whole-policy LLM"] = METHOD
    notice: Literal["LLM output is not legal advice and is not ground truth."] = NOT_LEGAL_ADVICE
    score: float
    points: float
    band: Literal["high", "mid", "low"]
    criteria: list[CriterionDisplay]


def build_display(review: PolicyReview, config: AppConfig) -> SinglePolicyDisplay:
    """Build the single-policy screen from a review. Feedback is omitted for Yes."""
    catalog = load_criteria(config.paths.criteria_path)
    names = {item.id: (item.display_order, item.name) for item in catalog.criteria}
    suggestions = {item.criterion_id: item.suggestion for item in review.feedback}
    rows: list[CriterionDisplay] = []
    for result in review.judgement.verdicts:
        criterion_id = result.judgement.criterion_id
        order, name = names[criterion_id]
        verdict = result.judgement.verdict
        rows.append(
            CriterionDisplay(
                order=order,
                criterion_id=criterion_id,
                name=name,
                condition_met=verdict,
                relevant_text=list(result.judgement.evidence_quotes),
                feedback=suggestions.get(criterion_id) if is_gap(verdict) else None,
            )
        )
    rows.sort(key=lambda item: item.order)
    return SinglePolicyDisplay(
        score=review.score.score,
        points=review.score.points,
        band=review.score.band,
        criteria=rows,
    )


def sample_review(config: AppConfig) -> PolicyReview:
    """A fixed review so the screen can be shown without calling the model."""
    catalog = load_criteria(config.paths.criteria_path)
    ordered = sorted(catalog.criteria, key=lambda item: item.display_order)
    verdicts: list[ModelResult] = []
    feedback: list[CriterionFeedback] = []
    citations: list[CitationReport] = []
    for index, criterion in enumerate(ordered):
        if index == 0:
            verdict: Literal["Yes", "Somewhat", "No"] = "Somewhat"
            quotes = ["We ask for consent before sending marketing email."]
            suggestion = "Describe how consent can be withdrawn."
        elif index == 1:
            verdict = "No"
            quotes = ["Contact privacy@example.com."]
            suggestion = "Name the purposes of processing."
        elif index == 2:
            verdict = "Yes"
            quotes = []
            suggestion = None
        else:
            verdict = "Yes"
            quotes = ["We keep records of processing."]
            suggestion = None
        verdicts.append(
            ModelResult(
                judgement=CriterionJudgement(
                    criterion_id=criterion.id,
                    verdict=verdict,
                    justification="Sample judgement for the single-policy view.",
                    evidence_quotes=quotes,
                ),
                model_id=config.model.id,
                usage=TokenUse(input_tokens=1, output_tokens=1),
                prompt_id="whole_policy_judge",
                prompt_version="v1",
            )
        )
        if suggestion is not None:
            feedback.append(
                CriterionFeedback(
                    criterion_id=criterion.id,
                    suggestion=suggestion,
                    evidence_quotes=quotes,
                )
            )
            citations.append(
                CitationReport(
                    criterion_id=criterion.id,
                    citation_support="supported",
                    unsupported_quotes=[],
                )
            )
    judgement = PolicyJudgement(
        prompt_id="whole_policy_judge",
        prompt_version="v1",
        model_id=config.model.id,
        verdicts=verdicts,
    )
    return PolicyReview(
        judgement=judgement,
        feedback=feedback,
        citations=citations,
        score=score_verdicts(judgement, config),
    )

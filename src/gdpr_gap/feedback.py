"""Suggest fixes for gap verdicts without changing those verdicts.

Feedback is a second model call. It runs only for verdicts that is_gap treats
as gaps. Yes criteria are skipped. Scores are computed from the verdicts alone.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gdpr_gap.config import AppConfig
from gdpr_gap.criteria import load_criteria
from gdpr_gap.gaps import is_gap
from gdpr_gap.judge import PolicyJudgement, _require_within_limit, judge_policy
from gdpr_gap.llm import LLMClient, ModelError
from gdpr_gap.logging import log_event
from gdpr_gap.prompts import PromptRegistry
from gdpr_gap.result_cache import ResultCache
from gdpr_gap.scoring import PolicyScore, score_policy

FEEDBACK_PROMPT_ID = "feedback"


class CriterionFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    suggestion: str = Field(min_length=1)
    evidence_quotes: list[str]


class CitationReport(BaseModel):
    """Whether each feedback quote appears in the policy."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    citation_support: Literal["supported", "unsupported"]
    unsupported_quotes: list[str]


class PolicyReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgement: PolicyJudgement
    feedback: list[CriterionFeedback]
    citations: list[CitationReport]
    score: PolicyScore


def evaluate_policy(
    policy_text: str,
    config: AppConfig,
    client: LLMClient,
    *,
    feedback: bool = True,
    cache: ResultCache | None = None,
    logger: logging.Logger | None = None,
) -> PolicyReview:
    """Judge every criterion, then optionally suggest fixes for gaps."""
    judgement = judge_policy(policy_text, config, client, cache=cache, logger=logger)
    items = suggest_fixes(policy_text, judgement, config, client, logger=logger) if feedback else []
    return PolicyReview(
        judgement=judgement,
        feedback=items,
        citations=check_citations(policy_text, items, logger=logger),
        score=score_verdicts(judgement, config),
    )


def suggest_fixes(
    policy_text: str,
    judgement: PolicyJudgement,
    config: AppConfig,
    client: LLMClient,
    *,
    logger: logging.Logger | None = None,
) -> list[CriterionFeedback]:
    """Return one suggestion per gap. The judgement object is not modified."""
    criteria = load_criteria(config.paths.criteria_path).criteria
    catalog = {item.id: item.condition for item in criteria}
    prompt = PromptRegistry(config.paths.prompts_dir).load(FEEDBACK_PROMPT_ID)
    fixes: list[CriterionFeedback] = []
    for result in judgement.verdicts:
        verdict = result.judgement.verdict
        if not is_gap(verdict):
            continue
        criterion_id = result.judgement.criterion_id
        rendered = PromptRegistry(config.paths.prompts_dir).render(
            prompt.prompt_id,
            prompt.version,
            criterion_id=criterion_id,
            condition=catalog[criterion_id],
            policy_text=policy_text,
            verdict=verdict,
            justification=result.judgement.justification,
        )
        _require_within_limit(rendered.text, policy_text, config, logger)
        text, _usage, _model_id = client.complete(
            rendered.text,
            prompt_id=rendered.prompt_id,
            prompt_version=rendered.version,
            model_id=config.model.id,
        )
        fixes.append(_parse_feedback(text, criterion_id))
    return fixes


def check_citations(
    policy_text: str,
    feedback: list[CriterionFeedback],
    *,
    logger: logging.Logger | None = None,
) -> list[CitationReport]:
    """Flag feedback quotes that do not appear in the policy. Verdicts are not involved."""
    reports: list[CitationReport] = []
    for item in feedback:
        unsupported = [
            quote for quote in item.evidence_quotes if not quote_in_policy(policy_text, quote)
        ]
        supported = bool(item.evidence_quotes) and not unsupported
        report = CitationReport(
            criterion_id=item.criterion_id,
            citation_support="supported" if supported else "unsupported",
            unsupported_quotes=unsupported,
        )
        if logger is not None and report.citation_support == "unsupported":
            log_event(
                logger,
                "citation_unsupported",
                criterion_id=item.criterion_id,
                unsupported_count=len(unsupported),
                quote_count=len(item.evidence_quotes),
            )
        reports.append(report)
    return reports


def quote_in_policy(policy_text: str, quote: str) -> bool:
    """Return whether a non-empty quote appears exactly in the policy."""
    text = quote.strip()
    return bool(text) and text in policy_text


def score_verdicts(judgement: PolicyJudgement, config: AppConfig) -> PolicyScore:
    """Score the recorded verdicts. Feedback is not an input."""
    catalog = load_criteria(config.paths.criteria_path)
    verdicts = {item.judgement.criterion_id: item.judgement.verdict for item in judgement.verdicts}
    weights = {item.id: item.weight for item in catalog.criteria}
    return score_policy(verdicts, weights, config.scoring)


def _parse_feedback(text: str, criterion_id: str) -> CriterionFeedback:
    payload = _json_object(text)
    try:
        parsed = CriterionFeedback.model_validate(payload)
    except ValidationError as exc:
        raise ModelError(f"Model response does not match the feedback schema: {exc}") from exc
    if parsed.criterion_id != criterion_id:
        raise ModelError(f"Feedback returned {parsed.criterion_id}, expected {criterion_id}.")
    return parsed


def _json_object(text: str) -> object:
    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[-1].strip() == "```":
            body = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ModelError("Feedback response must be a JSON object.") from exc

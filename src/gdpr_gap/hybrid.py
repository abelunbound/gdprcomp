"""Hybrid comparator: retrieved passages, then the LLM.

A criterion whose best passage is below the calibrated threshold is No.
That miss is logged on its own and the model is not called. The whole-policy
judge is a separate method and is not gated by this result.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gdpr_gap.config import AppConfig
from gdpr_gap.criteria import Criterion, load_criteria
from gdpr_gap.embeddings import EmbeddingClient
from gdpr_gap.judge import WHOLE_POLICY_PROMPT_ID, PolicyJudgement, _require_within_limit
from gdpr_gap.llm import CriterionJudgement, LLMClient, ModelResult, TokenUse
from gdpr_gap.logging import log_event
from gdpr_gap.prompts import PromptRegistry
from gdpr_gap.result_cache import ResultCache, make_cache_key
from gdpr_gap.retrieval import RankedPassage, retrieve_baseline

_MISS = "No passage met the retrieval threshold, so the model was not called."


class HybridError(Exception):
    """Raised when the hybrid comparator cannot finish a policy."""


class HybridJudgement(BaseModel):
    """Whole-policy-shaped verdicts plus the retrieval misses that skipped the model."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["hybrid"] = "hybrid"
    threshold: float = Field(ge=0, le=1)
    embedding_model_name: str = Field(min_length=1)
    embedding_model_version: str = Field(min_length=1)
    retrieval_misses: list[str]
    judgement: PolicyJudgement


def judge_hybrid(
    policy_text: str,
    config: AppConfig,
    embeddings: EmbeddingClient,
    client: LLMClient,
    *,
    cache: ResultCache | None = None,
    logger: logging.Logger | None = None,
) -> HybridJudgement:
    """Judge one policy from passages at or above the calibrated threshold.

    The policy text is the same input the whole-policy judge accepts. Only the
    passages that meet the threshold are sent to the model.
    """
    threshold = config.thresholds.retrieval_similarity
    if threshold is None:
        raise HybridError("Hybrid retrieval requires a calibrated retrieval_similarity threshold.")
    baseline = retrieve_baseline(policy_text, config, embeddings, logger=logger)
    prompt = PromptRegistry(config.paths.prompts_dir).load(WHOLE_POLICY_PROMPT_ID)
    criteria = sorted(
        load_criteria(config.paths.criteria_path).criteria,
        key=lambda item: item.display_order,
    )
    verdicts: list[ModelResult] = []
    misses: list[str] = []
    for criterion, retrieved in zip(criteria, baseline.criteria, strict=True):
        if retrieved.criterion_id != criterion.id:
            raise HybridError("Retrieval rows are not in criterion order.")
        relevant = [item for item in retrieved.passages if item.similarity >= threshold]
        if not relevant:
            misses.append(criterion.id)
            if logger is not None:
                log_event(
                    logger,
                    "retrieval_miss",
                    criterion_id=criterion.id,
                    similarity=retrieved.passages[0].similarity,
                    threshold=threshold,
                )
            verdicts.append(_miss(criterion.id, config.model.id, prompt.prompt_id, prompt.version))
            continue
        verdicts.append(
            _judge_passages(
                passages=relevant,
                criterion=criterion,
                prompt_id=prompt.prompt_id,
                prompt_version=prompt.version,
                config=config,
                client=client,
                cache=cache,
            )
        )
    judgement = PolicyJudgement(
        prompt_id=prompt.prompt_id,
        prompt_version=prompt.version,
        model_id=config.model.id,
        verdicts=verdicts,
    )
    if logger is not None:
        log_event(
            logger,
            "hybrid_judged",
            retrieval_miss_count=len(misses),
            model_call_count=len(criteria) - len(misses),
            threshold=threshold,
            embedding_model_version=baseline.model_version,
        )
    return HybridJudgement(
        threshold=threshold,
        embedding_model_name=baseline.model_name,
        embedding_model_version=baseline.model_version,
        retrieval_misses=misses,
        judgement=judgement,
    )


def _miss(criterion_id: str, model_id: str, prompt_id: str, prompt_version: str) -> ModelResult:
    return ModelResult(
        judgement=CriterionJudgement(
            criterion_id=criterion_id,
            verdict="No",
            justification=_MISS,
            evidence_quotes=[],
        ),
        model_id=model_id,
        usage=TokenUse(input_tokens=0, output_tokens=0, cost_usd=None),
        prompt_id=prompt_id,
        prompt_version=prompt_version,
    )


def _judge_passages(
    *,
    passages: list[RankedPassage],
    criterion: Criterion,
    prompt_id: str,
    prompt_version: str,
    config: AppConfig,
    client: LLMClient,
    cache: ResultCache | None,
) -> ModelResult:
    passage_text = "\n\n".join(item.text for item in passages)
    rendered = PromptRegistry(config.paths.prompts_dir).render(
        prompt_id,
        prompt_version,
        criterion_id=criterion.id,
        condition=criterion.condition,
        policy_text=passage_text,
    )
    if passage_text not in rendered.text:
        raise HybridError("The prompt does not contain the retrieved passages.")
    _require_within_limit(rendered.text, passage_text, config, None)

    def produce() -> ModelResult:
        result = client.judge(
            rendered.text,
            prompt_id=rendered.prompt_id,
            prompt_version=rendered.version,
            model_id=config.model.id,
        )
        if result.judgement.criterion_id != criterion.id:
            raise HybridError(
                f"Model returned {result.judgement.criterion_id}, expected {criterion.id}."
            )
        return result

    if cache is None:
        return produce()
    key = make_cache_key(
        policy_text=passage_text,
        criterion_id=criterion.id,
        prompt_id=rendered.prompt_id,
        prompt_version=rendered.version,
        model_id=config.model.id,
    )
    return cache.judgement(key, produce)

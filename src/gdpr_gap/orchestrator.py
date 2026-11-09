"""One entry point for the app and for batch runs.

The method is chosen here: whole_policy_llm is the default, hybrid and
legal_bert_only are comparators. Every result records that method.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gdpr_gap.config import AppConfig
from gdpr_gap.embeddings import EmbeddingClient
from gdpr_gap.feedback import PolicyReview, evaluate_policy
from gdpr_gap.hybrid import HybridJudgement, judge_hybrid
from gdpr_gap.legal_bert import LegalBertBaseline, score_legal_bert
from gdpr_gap.llm import LLMClient
from gdpr_gap.logging import log_event
from gdpr_gap.result_cache import ResultCache

DEFAULT_METHOD: Literal["whole_policy_llm"] = "whole_policy_llm"
Method = Literal["whole_policy_llm", "hybrid", "legal_bert_only"]
_METHODS: tuple[Method, ...] = ("whole_policy_llm", "hybrid", "legal_bert_only")


class OrchestratorError(Exception):
    """Raised when a run cannot be dispatched."""


class PolicyDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    text: str


class PolicyRun(BaseModel):
    """One policy scored by one method. The method field is always set."""

    model_config = ConfigDict(extra="forbid")

    method: Method
    policy_id: str = Field(min_length=1)
    review: PolicyReview | None = None
    hybrid: HybridJudgement | None = None
    retrieval: LegalBertBaseline | None = None

    @model_validator(mode="after")
    def payload_matches_method(self) -> PolicyRun:
        if self.method == "whole_policy_llm":
            if self.review is None or self.hybrid is not None or self.retrieval is not None:
                raise ValueError("whole_policy_llm records a review and nothing else")
        elif self.method == "hybrid":
            if self.hybrid is None or self.review is not None or self.retrieval is not None:
                raise ValueError("hybrid records a hybrid judgement and nothing else")
            if self.hybrid.method != "hybrid":
                raise ValueError("hybrid result method does not match the run")
        elif self.retrieval is None or self.review is not None or self.hybrid is not None:
            raise ValueError("legal_bert_only records a retrieval signal and nothing else")
        elif self.retrieval.method != "legal_bert_only":
            raise ValueError("retrieval result method does not match the run")
        return self


def run_policy(
    policy_text: str,
    config: AppConfig,
    method: str = DEFAULT_METHOD,
    *,
    policy_id: str,
    client: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
    cache: ResultCache | None = None,
    feedback: bool = True,
    logger: logging.Logger | None = None,
) -> PolicyRun:
    """Score one policy with the selected method. The default is whole_policy_llm."""
    selected = _method(method)
    result = _dispatch(
        policy_text,
        config,
        selected,
        policy_id=policy_id,
        client=client,
        embeddings=embeddings,
        cache=cache,
        feedback=feedback,
        logger=logger,
    )
    if logger is not None:
        log_event(logger, "policy_run", method=result.method, policy_id=result.policy_id)
    return result


def run_batch(
    policies: list[PolicyDocument],
    config: AppConfig,
    method: str = DEFAULT_METHOD,
    *,
    client: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
    cache: ResultCache | None = None,
    feedback: bool = True,
    logger: logging.Logger | None = None,
) -> list[PolicyRun]:
    """Score every policy with the same method, through run_policy."""
    selected = _method(method)
    if not policies:
        raise OrchestratorError("Batch has no policies.")
    ids = [item.policy_id for item in policies]
    if len(set(ids)) != len(ids):
        raise OrchestratorError("Each policy id in a batch must be unique.")
    return [
        run_policy(
            item.text,
            config,
            selected,
            policy_id=item.policy_id,
            client=client,
            embeddings=embeddings,
            cache=cache,
            feedback=feedback,
            logger=logger,
        )
        for item in policies
    ]


def whole_policy_result(policy_id: str, review: PolicyReview) -> PolicyRun:
    """Record a whole-policy review. The app uses this after its stepped judge."""
    return PolicyRun(method="whole_policy_llm", policy_id=policy_id, review=review)


def _dispatch(
    policy_text: str,
    config: AppConfig,
    method: Method,
    *,
    policy_id: str,
    client: LLMClient | None,
    embeddings: EmbeddingClient | None,
    cache: ResultCache | None,
    feedback: bool,
    logger: logging.Logger | None,
) -> PolicyRun:
    if method == "whole_policy_llm":
        if client is None:
            raise OrchestratorError("whole_policy_llm requires an LLM client.")
        review = evaluate_policy(
            policy_text,
            config,
            client,
            feedback=feedback,
            cache=cache,
            logger=logger,
        )
        return whole_policy_result(policy_id, review)
    if method == "hybrid":
        if client is None or embeddings is None:
            raise OrchestratorError("hybrid requires an LLM client and an embedding client.")
        return PolicyRun(
            method="hybrid",
            policy_id=policy_id,
            hybrid=judge_hybrid(
                policy_text,
                config,
                embeddings,
                client,
                cache=cache,
                logger=logger,
            ),
        )
    if embeddings is None:
        raise OrchestratorError("legal_bert_only requires an embedding client.")
    return PolicyRun(
        method="legal_bert_only",
        policy_id=policy_id,
        retrieval=score_legal_bert(policy_text, config, embeddings, logger=logger),
    )


def _method(value: str) -> Method:
    if value in _METHODS:
        return value
    raise OrchestratorError("Unknown method. Choose whole_policy_llm, hybrid, or legal_bert_only.")

"""Legal-BERT-only comparator.

The result is a similarity ranking. It is a baseline retrieval signal, not an
LLM verdict, and it has no Yes, Somewhat, or No field. The whole-policy judge
does not read it.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gdpr_gap.config import AppConfig
from gdpr_gap.embeddings import EmbeddingClient
from gdpr_gap.retrieval import RankedPassage, retrieve_baseline

RETRIEVAL_SIGNAL: Literal["baseline/retrieval signal"] = "baseline/retrieval signal"


class LegalBertCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    best_similarity: float
    passages: list[RankedPassage] = Field(min_length=1)

    @model_validator(mode="after")
    def best_is_the_first_passage(self) -> LegalBertCriterion:
        if self.passages[0].similarity != self.best_similarity:
            raise ValueError("best_similarity must be the first ranked passage")
        return self


class LegalBertBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["legal_bert_only"] = "legal_bert_only"
    signal: Literal["baseline/retrieval signal"] = RETRIEVAL_SIGNAL
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    criteria: list[LegalBertCriterion] = Field(min_length=22, max_length=22)

    @model_validator(mode="after")
    def one_row_per_criterion(self) -> LegalBertBaseline:
        ids = [item.criterion_id for item in self.criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("each criterion must be scored once")
        return self


def score_legal_bert(
    policy_text: str,
    config: AppConfig,
    client: EmbeddingClient,
    *,
    logger: logging.Logger | None = None,
) -> LegalBertBaseline:
    """Rank passages for every criterion. No model verdict is produced."""
    baseline = retrieve_baseline(policy_text, config, client, logger=logger)
    return LegalBertBaseline(
        model_name=baseline.model_name,
        model_version=baseline.model_version,
        criteria=[
            LegalBertCriterion(
                criterion_id=row.criterion_id,
                best_similarity=row.passages[0].similarity,
                passages=row.passages,
            )
            for row in baseline.criteria
        ],
    )

"""Legal-BERT retrieval baseline for comparator methods.

Each passage is scored by cosine similarity against a criterion condition.
The score is topical relevance. It is not a compliance verdict and it is not
Yes, Somewhat, or No. The whole-policy judge does not read this output.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gdpr_gap.config import AppConfig
from gdpr_gap.criteria import load_criteria
from gdpr_gap.embeddings import EmbeddingClient, PassageEmbedding
from gdpr_gap.logging import log_event
from gdpr_gap.segmentation import Passage, segment_passages

TOPICAL_RELEVANCE: Literal["Topical relevance is not a compliance verdict."] = (
    "Topical relevance is not a compliance verdict."
)


class RetrievalError(Exception):
    """Raised when passage similarity cannot be scored."""


class RankedPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    start_word: int = Field(ge=0)
    end_word: int = Field(ge=0)
    text: str = Field(min_length=1)
    similarity: float


class CriterionRetrieval(BaseModel):
    """Passages ranked by topical similarity to one condition sentence."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    passages: list[RankedPassage] = Field(min_length=1)

    @model_validator(mode="after")
    def similarity_does_not_increase(self) -> CriterionRetrieval:
        scores = [item.similarity for item in self.passages]
        if scores != sorted(scores, reverse=True):
            raise ValueError("passages must be ranked from highest similarity")
        return self


class RetrievalBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: Literal["topical relevance"] = "topical relevance"
    description: Literal["Topical relevance is not a compliance verdict."] = TOPICAL_RELEVANCE
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    criteria: list[CriterionRetrieval] = Field(min_length=22, max_length=22)

    @model_validator(mode="after")
    def one_row_per_criterion(self) -> RetrievalBaseline:
        ids = [item.criterion_id for item in self.criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("each criterion must be retrieved once")
        return self


def retrieve_baseline(
    policy_text: str,
    config: AppConfig,
    client: EmbeddingClient,
    *,
    logger: logging.Logger | None = None,
) -> RetrievalBaseline:
    """Rank every passage against every criterion condition.

    Similarity is not thresholded. A high score does not mean the condition is met.
    """
    passages = segment_passages(policy_text, config.segmentation)
    if not passages:
        raise RetrievalError("Cannot retrieve passages from an empty policy.")
    criteria = sorted(
        load_criteria(config.paths.criteria_path).criteria,
        key=lambda item: item.display_order,
    )
    embedded = client.embed(
        [passage.text for passage in passages] + [item.condition for item in criteria]
    )
    passage_vectors = embedded[: len(passages)]
    condition_vectors = embedded[len(passages) :]
    model_name, model_version = _one_model(passage_vectors + condition_vectors)
    rows = [
        CriterionRetrieval(
            criterion_id=criterion.id,
            passages=_rank(passages, passage_vectors, condition),
        )
        for criterion, condition in zip(criteria, condition_vectors, strict=True)
    ]
    baseline = RetrievalBaseline(
        model_name=model_name,
        model_version=model_version,
        criteria=rows,
    )
    if logger is not None:
        log_event(
            logger,
            "retrieval_baseline",
            model_name=model_name,
            model_version=model_version,
            passage_count=len(passages),
            criterion_count=len(rows),
        )
    return baseline


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return the cosine of two equal-length vectors."""
    if len(left) != len(right) or not left:
        raise RetrievalError("Vectors must be non-empty and the same length.")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise RetrievalError("Cannot score a zero vector.")
    return dot / (left_norm * right_norm)


def _rank(
    passages: list[Passage],
    vectors: list[PassageEmbedding],
    condition: PassageEmbedding,
) -> list[RankedPassage]:
    scored = [
        RankedPassage(
            index=passage.index,
            start_word=passage.start_word,
            end_word=passage.end_word,
            text=passage.text,
            similarity=cosine_similarity(vector.vector, condition.vector),
        )
        for passage, vector in zip(passages, vectors, strict=True)
    ]
    scored.sort(key=lambda item: (-item.similarity, item.index))
    return scored


def _one_model(embeddings: list[PassageEmbedding]) -> tuple[str, str]:
    names = {item.model_name for item in embeddings}
    versions = {item.model_version for item in embeddings}
    if len(names) != 1 or len(versions) != 1:
        raise RetrievalError("Embeddings in one baseline must share a model name and version.")
    return names.pop(), versions.pop()

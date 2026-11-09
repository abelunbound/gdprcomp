"""Split policy text into overlapping passages for comparator methods only.

The whole-policy LLM judge reads the full cleaned text. It must not be gated,
filtered, or skipped because of these passages. Hybrid retrieval and Legal-BERT
are the only callers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gdpr_gap.config import SegmentationConfig


class SegmentationError(Exception):
    """Raised when passage settings cannot build a forward-moving window."""


class Passage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    start_word: int = Field(ge=0)
    end_word: int = Field(ge=0)
    text: str
    use: Literal["comparator"] = "comparator"


def segment_passages(text: str, segmentation: SegmentationConfig) -> list[Passage]:
    """Cut whitespace-separated words into overlapping comparator passages.

    The last passage is shorter when the tail does not fill a window. Every
    word, including that tail, belongs to at least one passage.
    """
    if segmentation.overlap_words >= segmentation.passage_words:
        raise SegmentationError("overlap_words must be less than passage_words")
    words = text.split()
    if not words:
        return []
    stride = segmentation.passage_words - segmentation.overlap_words
    passages: list[Passage] = []
    start = 0
    while start < len(words):
        end = min(start + segmentation.passage_words, len(words))
        passages.append(
            Passage(
                index=len(passages),
                start_word=start,
                end_word=end,
                text=" ".join(words[start:end]),
            )
        )
        if end == len(words):
            break
        start += stride
    return passages


def passages_cover(text: str, passages: list[Passage]) -> bool:
    """Return whether every word, including the tail, sits in some passage."""
    words = text.split()
    if not words:
        return passages == []
    if not passages:
        return False
    covered = [False] * len(words)
    for passage in passages:
        for index in range(passage.start_word, passage.end_word):
            if index >= len(words):
                return False
            covered[index] = True
    return all(covered) and passages[-1].end_word == len(words)

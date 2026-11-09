"""Legal-BERT embeddings for the comparator methods.

Mean pooling is used. Padding tokens are excluded. The same text and model
revision always produce the same vector. This module does not download model
weights unless load_legal_bert is called with confirm=True. The whole-policy
judge does not use these embeddings.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from gdpr_gap.config import EmbeddingConfig
from gdpr_gap.logging import log_event

LEGAL_BERT_NAME = "nlpaueb/legal-bert-base-uncased"
_CACHE_DIR = Path("data/models")


class EmbeddingError(Exception):
    """Raised when text cannot be embedded or the configured model cannot be loaded."""


class PassageEmbedding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector: list[float] = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)


class HiddenStates:
    """Token vectors and a 1/0 mask. A 0 marks a padding token."""

    def __init__(self, states: list[list[list[float]]], attention_mask: list[list[int]]) -> None:
        self.states = states
        self.attention_mask = attention_mask


class TextEncoder(Protocol):
    name: str
    version: str

    def encode(self, texts: list[str]) -> HiddenStates:
        """Return one hidden-state row per text. Do not truncate silently."""


class EmbeddingClient:
    """Embed a batch of texts with the configured Legal-BERT revision."""

    def __init__(
        self,
        settings: EmbeddingConfig,
        encoder: TextEncoder,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        if encoder.name != settings.model_name or encoder.version != settings.revision:
            raise EmbeddingError(
                "Encoder does not match embedding.model_name and embedding.revision."
            )
        self._settings = settings
        self._encoder = encoder
        self._logger = logger

    def embed(self, texts: list[str]) -> list[PassageEmbedding]:
        """Return one mean-pooled vector per text, in the same order."""
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Cannot embed empty text.")
        embeddings: list[PassageEmbedding] = []
        batch_size = self._settings.batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            hidden = self._encoder.encode(batch)
            for vector in mean_pool(hidden):
                embeddings.append(
                    PassageEmbedding(
                        vector=vector,
                        model_name=self._encoder.name,
                        model_version=self._encoder.version,
                    )
                )
        if self._logger is not None:
            dimension = len(embeddings[0].vector) if embeddings else 0
            log_event(
                self._logger,
                "embedding_call",
                model_name=self._encoder.name,
                model_version=self._encoder.version,
                text_count=len(texts),
                dimension=dimension,
            )
        return embeddings


def mean_pool(hidden: HiddenStates) -> list[list[float]]:
    """Average token vectors, ignoring positions whose attention mask is 0."""
    if len(hidden.states) != len(hidden.attention_mask):
        raise EmbeddingError("Hidden states and attention mask have different batch sizes.")
    pooled: list[list[float]] = []
    for tokens, mask in zip(hidden.states, hidden.attention_mask, strict=True):
        if len(tokens) != len(mask):
            raise EmbeddingError("A hidden-state row does not match its attention mask.")
        if not tokens:
            raise EmbeddingError("A hidden-state row has no tokens.")
        width = len(tokens[0])
        totals = [0.0] * width
        kept = 0
        for token, keep in zip(tokens, mask, strict=True):
            if len(token) != width:
                raise EmbeddingError("Token vectors in one row have different widths.")
            if keep == 0:
                continue
            kept += 1
            for index, value in enumerate(token):
                totals[index] += value
        if kept == 0:
            raise EmbeddingError("A row has no unmasked tokens to pool.")
        pooled.append([value / kept for value in totals])
    return pooled


def load_legal_bert(settings: EmbeddingConfig, *, confirm: bool = False) -> TextEncoder:
    """Load Legal-BERT. Refuses, and writes nothing, unless confirm is True."""
    if settings.model_name != LEGAL_BERT_NAME:
        raise EmbeddingError(f"Unsupported embedding model: {settings.model_name}")
    if not confirm:
        raise EmbeddingError(
            "Legal-BERT download was not confirmed. "
            f"{settings.model_name} at revision {settings.revision} was not fetched."
        )
    return _load_confirmed_model(settings)


def _load_confirmed_model(settings: EmbeddingConfig) -> TextEncoder:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise EmbeddingError("torch and transformers are required to load Legal-BERT.") from exc
    cache_dir = _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    modules = cast(Any, transformers)
    tokenizer = modules.AutoTokenizer.from_pretrained(
        settings.model_name,
        revision=settings.revision,
        cache_dir=str(cache_dir),
    )
    model = modules.AutoModel.from_pretrained(
        settings.model_name,
        revision=settings.revision,
        cache_dir=str(cache_dir),
    )
    model.eval()
    return _TransformerEncoder(settings, tokenizer, model, cast(Any, torch))


class _TransformerEncoder:
    def __init__(
        self,
        settings: EmbeddingConfig,
        tokenizer: Any,
        model: Any,
        torch: Any,
    ) -> None:
        self.name = settings.model_name
        self.version = settings.revision
        self._settings = settings
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch

    def encode(self, texts: list[str]) -> HiddenStates:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        width = int(encoded["input_ids"].shape[1])
        if width > self._settings.max_length:
            raise EmbeddingError(
                "Input exceeds the embedding model limit and was not truncated: "
                f"{width} tokens > {self._settings.max_length}."
            )
        with self._torch.no_grad():
            output = self._model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        states = cast(list[list[list[float]]], output.last_hidden_state.detach().cpu().tolist())
        mask = cast(list[list[int]], encoded["attention_mask"].detach().cpu().tolist())
        return HiddenStates(states, mask)

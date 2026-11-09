"""Cache LLM judgements and embedding vectors.

A cache entry is identified only by the policy-text hash, criterion, prompt id,
prompt version, and model. Timeouts, batch size, weights, seeds, and the rest
of the config are not part of the key. The prompt body is not part of the key
either: a changed prompt needs a new prompt version.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import overload

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gdpr_gap.embeddings import PassageEmbedding
from gdpr_gap.llm import ModelResult
from gdpr_gap.logging import log_event

Kind = str


class CacheError(Exception):
    """Raised when a cache entry cannot be stored or read."""


class CacheKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    prompt_id: str = Field(pattern=r"^[a-z0-9_]+$")
    prompt_version: str = Field(pattern=r"^v[1-9][0-9]*$")
    model_id: str = Field(min_length=1)


class _StoredJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: CacheKey
    result: ModelResult


class _StoredEmbedding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: CacheKey
    result: PassageEmbedding


class ResultCache:
    """Store and reuse judgements and embeddings."""

    def __init__(self, directory: str | Path, *, logger: logging.Logger | None = None) -> None:
        self._directory = Path(directory)
        self._logger = logger

    def judgement(self, key: CacheKey, produce: Callable[[], ModelResult]) -> ModelResult:
        """Return the stored judgement, or store the result of produce on a miss."""
        found = self.get_judgement(key)
        if found is not None:
            return found
        result = produce()
        self.put_judgement(key, result)
        return result

    def embedding(self, key: CacheKey, produce: Callable[[], PassageEmbedding]) -> PassageEmbedding:
        """Return the stored embedding, or store the result of produce on a miss."""
        found = self.get_embedding(key)
        if found is not None:
            return found
        result = produce()
        self.put_embedding(key, result)
        return result

    def get_judgement(self, key: CacheKey) -> ModelResult | None:
        """Return a stored judgement. A missing entry is a logged miss."""
        stored = self._read(key, "judgement", _StoredJudgement)
        if stored is None:
            return None
        return stored.result

    def put_judgement(self, key: CacheKey, result: ModelResult) -> None:
        """Store a judgement under its identity key."""
        if result.model_id != key.model_id or result.judgement.criterion_id != key.criterion_id:
            raise CacheError("Judgement does not match the cache key model and criterion.")
        self._write(key, "judgement", _StoredJudgement(key=key, result=result))

    def get_embedding(self, key: CacheKey) -> PassageEmbedding | None:
        """Return a stored embedding. A missing entry is a logged miss."""
        stored = self._read(key, "embedding", _StoredEmbedding)
        if stored is None:
            return None
        return stored.result

    def put_embedding(self, key: CacheKey, result: PassageEmbedding) -> None:
        """Store an embedding under its identity key."""
        model_id = embedding_model_id(result.model_name, result.model_version)
        if key.model_id != model_id:
            raise CacheError("Embedding does not match the cache key model.")
        self._write(key, "embedding", _StoredEmbedding(key=key, result=result))

    @overload
    def _read(
        self,
        key: CacheKey,
        kind: Kind,
        model: type[_StoredJudgement],
    ) -> _StoredJudgement | None: ...

    @overload
    def _read(
        self,
        key: CacheKey,
        kind: Kind,
        model: type[_StoredEmbedding],
    ) -> _StoredEmbedding | None: ...

    def _read(
        self,
        key: CacheKey,
        kind: Kind,
        model: type[_StoredJudgement] | type[_StoredEmbedding],
    ) -> _StoredJudgement | _StoredEmbedding | None:
        path = self._path(kind, key)
        if not path.is_file():
            self._log("cache_miss", key, kind)
            return None
        try:
            stored = model.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise CacheError(f"Cache entry is unreadable: {path}") from exc
        if stored.key != key:
            raise CacheError(f"Cache entry key does not match the requested key: {path}")
        self._log("cache_hit", key, kind)
        return stored

    def _write(self, key: CacheKey, kind: Kind, stored: BaseModel) -> None:
        path = self._path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(stored.model_dump_json(), encoding="utf-8")
        temporary.replace(path)

    def _path(self, kind: Kind, key: CacheKey) -> Path:
        identity = json.dumps(
            {"kind": kind, **key.model_dump()},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self._directory / kind / f"{digest}.json"

    def _log(self, event: str, key: CacheKey, kind: Kind) -> None:
        if self._logger is None:
            return
        log_event(
            self._logger,
            event,
            kind=kind,
            policy_hash=key.policy_hash,
            criterion_id=key.criterion_id,
            prompt_id=key.prompt_id,
            prompt_version=key.prompt_version,
            model_id=key.model_id,
        )


def make_cache_key(
    *,
    policy_text: str,
    criterion_id: str,
    prompt_id: str,
    prompt_version: str,
    model_id: str,
) -> CacheKey:
    """Build a cache key. The policy text is hashed and is not stored in the key."""
    return CacheKey(
        policy_hash=hashlib.sha256(policy_text.encode("utf-8")).hexdigest(),
        criterion_id=criterion_id,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        model_id=model_id,
    )


def embedding_model_id(model_name: str, model_version: str) -> str:
    """Return the model identity stored in an embedding cache key."""
    return f"{model_name}@{model_version}"

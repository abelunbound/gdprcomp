import inspect
import json
import logging
from pathlib import Path
from typing import Any, cast

import pytest

from gdpr_gap.config import load_config
from gdpr_gap.embeddings import PassageEmbedding
from gdpr_gap.llm import CriterionJudgement, ModelResult, TokenUse
from gdpr_gap.result_cache import (
    CacheError,
    CacheKey,
    ResultCache,
    embedding_model_id,
    make_cache_key,
)

POLICY = "We ask for consent before sending marketing email."
CRITERION = "consent_mechanisms"
PROMPT_ID = "whole_policy_judge"
PROMPT_VERSION = "v1"
MODEL_ID = "claude-haiku-5-5"


def _logger() -> tuple[logging.Logger, list[logging.LogRecord]]:
    logger = logging.getLogger("gdpr_gap_result_cache_test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger.addHandler(Capture())
    return logger, records


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    return cast(dict[str, Any], getattr(record, "fields"))


def _judgement() -> ModelResult:
    return ModelResult(
        judgement=CriterionJudgement(
            criterion_id=CRITERION,
            verdict="Yes",
            justification="The sentence says consent is requested.",
            evidence_quotes=[POLICY],
        ),
        model_id=MODEL_ID,
        usage=TokenUse(input_tokens=10, output_tokens=4, cost_usd=None),
        prompt_id=PROMPT_ID,
        prompt_version=PROMPT_VERSION,
    )


def _key(**overrides: str) -> CacheKey:
    values = {
        "policy_text": POLICY,
        "criterion_id": CRITERION,
        "prompt_id": PROMPT_ID,
        "prompt_version": PROMPT_VERSION,
        "model_id": MODEL_ID,
    }
    values.update(overrides)
    return make_cache_key(**values)


def test_cache_key_uses_only_the_identity_fields() -> None:
    parameters = tuple(inspect.signature(make_cache_key).parameters)
    assert parameters == (
        "policy_text",
        "criterion_id",
        "prompt_id",
        "prompt_version",
        "model_id",
    )

    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": "test-key-not-a-real-secret"},
        load_env_file=False,
    )
    changed = config.model_copy(
        update={
            "seed": config.seed + 1,
            "model": config.model.model_copy(update={"timeout_seconds": 10}),
            "embedding": config.embedding.model_copy(update={"batch_size": 1}),
        }
    )
    assert changed.seed != config.seed
    first = _key()
    second = _key()
    assert first == second
    assert set(first.model_dump()) == {
        "policy_hash",
        "criterion_id",
        "prompt_id",
        "prompt_version",
        "model_id",
    }
    assert POLICY not in json.dumps(first.model_dump())
    assert _key(prompt_id="feedback") != first
    assert _key(prompt_version="v2") != first
    assert _key(policy_text="A different policy.") != first
    assert _key(model_id="claude-sonnet-5-5") != first


def test_judgement_miss_then_hit_is_logged_and_computed_once(tmp_path: Path) -> None:
    logger, records = _logger()
    cache = ResultCache(tmp_path, logger=logger)
    calls = 0

    def produce() -> ModelResult:
        nonlocal calls
        calls += 1
        return _judgement()

    key = _key()
    first = cache.judgement(key, produce)
    second = cache.judgement(key, produce)

    assert first == second == _judgement()
    assert calls == 1
    assert [record.getMessage() for record in records] == ["cache_miss", "cache_hit"]
    assert _fields(records[0])["prompt_id"] == PROMPT_ID
    assert _fields(records[0])["prompt_version"] == PROMPT_VERSION
    assert _fields(records[0])["model_id"] == MODEL_ID
    assert POLICY not in json.dumps(_fields(records[0]))
    stored = json.loads(next((tmp_path / "judgement").glob("*.json")).read_text(encoding="utf-8"))
    assert "timeout_seconds" not in stored
    assert "batch_size" not in stored
    assert "seed" not in stored


def test_embedding_is_cached_separately_from_a_judgement(tmp_path: Path) -> None:
    logger, records = _logger()
    cache = ResultCache(tmp_path, logger=logger)
    model_id = embedding_model_id("nlpaueb/legal-bert-base-uncased", "a" * 40)
    key = _key(model_id=model_id)
    embedding = PassageEmbedding(
        vector=[0.25, 0.5],
        model_name="nlpaueb/legal-bert-base-uncased",
        model_version="a" * 40,
    )
    calls = 0

    def produce() -> PassageEmbedding:
        nonlocal calls
        calls += 1
        return embedding

    assert cache.embedding(key, produce) == embedding
    assert cache.embedding(key, produce) == embedding
    assert calls == 1
    assert cache.get_judgement(key) is None
    assert [record.getMessage() for record in records] == [
        "cache_miss",
        "cache_hit",
        "cache_miss",
    ]
    assert _fields(records[0])["kind"] == "embedding"
    assert _fields(records[2])["kind"] == "judgement"


def test_mismatched_result_is_not_stored(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path)
    wrong = _judgement().model_copy(update={"model_id": "claude-sonnet-5-5"})

    with pytest.raises(CacheError, match="model"):
        cache.put_judgement(_key(), wrong)

    assert list(tmp_path.rglob("*.json")) == []


def test_unreadable_entry_fails(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path)
    cache.put_judgement(_key(), _judgement())
    path = next((tmp_path / "judgement").glob("*.json"))
    path.write_text("{", encoding="utf-8")

    with pytest.raises(CacheError, match="unreadable"):
        cache.get_judgement(_key())

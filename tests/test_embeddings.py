import logging
from typing import Any, cast

import pytest

from gdpr_gap.config import EmbeddingConfig, load_config
from gdpr_gap.embeddings import (
    EmbeddingClient,
    EmbeddingError,
    HiddenStates,
    load_legal_bert,
    mean_pool,
)

REVISION = "15b570cbf88259610b082a167dacc190124f60f6"
NAME = "nlpaueb/legal-bert-base-uncased"


def _settings(batch_size: int = 8) -> EmbeddingConfig:
    return EmbeddingConfig(model_name=NAME, revision=REVISION, batch_size=batch_size)


class FixedEncoder:
    name = NAME
    version = REVISION

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode(self, texts: list[str]) -> HiddenStates:
        self.batches.append(list(texts))
        states: list[list[list[float]]] = []
        mask: list[list[int]] = []
        for text in texts:
            value = float(len(text))
            states.append([[value, value + 1.0], [99.0, 99.0]])
            mask.append([1, 0])
        return HiddenStates(states, mask)


def test_mean_pool_ignores_padding() -> None:
    hidden = HiddenStates(
        states=[[[1.0, 3.0], [10.0, 10.0]], [[1.0, 1.0], [3.0, 5.0]]],
        attention_mask=[[1, 0], [1, 1]],
    )

    assert mean_pool(hidden) == [[1.0, 3.0], [2.0, 3.0]]


def test_embeddings_are_deterministic_and_keep_order() -> None:
    encoder = FixedEncoder()
    client = EmbeddingClient(_settings(batch_size=2), encoder)
    texts = ["aa", "b", "cccc"]

    first = client.embed(texts)
    second = client.embed(texts)

    assert first == second
    assert [item.vector for item in first] == [[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]
    assert {item.model_name for item in first} == {NAME}
    assert {item.model_version for item in first} == {REVISION}
    assert encoder.batches == [["aa", "b"], ["cccc"], ["aa", "b"], ["cccc"]]


def test_model_version_is_logged() -> None:
    logger = logging.getLogger("gdpr_gap_embedding_test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger.addHandler(Capture())
    EmbeddingClient(_settings(), FixedEncoder(), logger=logger).embed(["consent"])

    assert len(records) == 1
    fields = cast(dict[str, Any], getattr(records[0], "fields"))
    assert fields["model_name"] == NAME
    assert fields["model_version"] == REVISION
    assert fields["text_count"] == 1
    assert "consent" not in str(fields)


def test_empty_text_is_rejected_before_encoding() -> None:
    encoder = FixedEncoder()
    client = EmbeddingClient(_settings(), encoder)

    with pytest.raises(EmbeddingError, match="empty"):
        client.embed(["kept", "  "])

    assert encoder.batches == []


def test_encoder_must_match_the_configured_revision() -> None:
    encoder = FixedEncoder()
    encoder.version = "a" * 40

    with pytest.raises(EmbeddingError, match="revision"):
        EmbeddingClient(_settings(), encoder)


def test_legal_bert_is_not_downloaded_without_confirmation() -> None:
    settings = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": "test-key-not-a-real-secret"},
        load_env_file=False,
    ).embedding

    with pytest.raises(EmbeddingError, match="not fetched"):
        load_legal_bert(settings)

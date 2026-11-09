import logging
from typing import Any, cast

import pytest

from gdpr_gap.config import AppConfig, EmbeddingConfig, SegmentationConfig, load_config
from gdpr_gap.criteria import load_criteria
from gdpr_gap.embeddings import EmbeddingClient, HiddenStates
from gdpr_gap.retrieval import RetrievalError, cosine_similarity, retrieve_baseline

NAME = "nlpaueb/legal-bert-base-uncased"
REVISION = "15b570cbf88259610b082a167dacc190124f60f6"
KEY = "test-key-not-a-real-secret"


class TextEncoder:
    name = NAME
    version = REVISION

    def __init__(self, marked: str | None) -> None:
        self.marked = marked
        self.calls = 0
        self.texts: list[str] = []

    def encode(self, texts: list[str]) -> HiddenStates:
        self.calls += 1
        self.texts.extend(texts)
        states: list[list[list[float]]] = []
        mask: list[list[int]] = []
        for text in texts:
            vector = [1.0, 0.0] if text == self.marked else [0.0, 1.0]
            states.append([vector])
            mask.append([1])
        return HiddenStates(states, mask)


class ZeroEncoder(TextEncoder):
    def encode(self, texts: list[str]) -> HiddenStates:
        self.calls += 1
        return HiddenStates([[[0.0, 0.0]] for _ in texts], [[1] for _ in texts])


def _config(passage_words: int) -> AppConfig:
    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )
    segmentation = SegmentationConfig(passage_words=passage_words, overlap_words=0)
    return config.model_copy(update={"segmentation": segmentation})


def _client(encoder: TextEncoder) -> EmbeddingClient:
    settings = EmbeddingConfig(model_name=NAME, revision=REVISION, batch_size=8)
    return EmbeddingClient(settings, encoder)


def test_ranked_passages_score_topical_similarity_and_record_the_model() -> None:
    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )
    ordered = sorted(
        load_criteria(config.paths.criteria_path).criteria,
        key=lambda item: item.display_order,
    )
    condition = ordered[0].condition
    assert " ".join(condition.split()) == condition
    tuned = _config(len(condition.split()))
    policy = f"{condition} unrelated filler words extra"
    encoder = TextEncoder(condition)

    baseline = retrieve_baseline(policy, tuned, _client(encoder))

    assert baseline.signal == "topical relevance"
    assert baseline.description == "Topical relevance is not a compliance verdict."
    assert baseline.model_name == NAME
    assert baseline.model_version == REVISION
    assert "verdict" not in baseline.model_dump()
    assert [row.criterion_id for row in baseline.criteria][0] == "data_processing_records"
    matched = baseline.criteria[0]
    assert [item.index for item in matched.passages] == [0, 1]
    assert matched.passages[0].similarity == 1
    assert matched.passages[1].similarity == 0
    other = baseline.criteria[1]
    assert [item.index for item in other.passages] == [1, 0]
    assert len(encoder.texts) == 24
    assert encoder.texts.count(condition) == 2


def test_equal_scores_keep_passage_order() -> None:
    config = _config(3)
    encoder = TextEncoder(marked=None)
    baseline = retrieve_baseline("one two three four five six", config, _client(encoder))

    assert [item.index for item in baseline.criteria[0].passages] == [0, 1]
    assert baseline.criteria[0].passages[0].similarity == 1


def test_empty_policy_and_zero_vector_are_rejected() -> None:
    config = _config(4)
    encoder = TextEncoder("unused")

    with pytest.raises(RetrievalError, match="empty"):
        retrieve_baseline("   ", config, _client(encoder))
    assert encoder.calls == 0

    with pytest.raises(RetrievalError, match="zero"):
        retrieve_baseline("one two three four", config, _client(ZeroEncoder(None)))


def test_retrieval_log_records_the_model_version_only() -> None:
    logger = logging.getLogger("gdpr_gap_retrieval_test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger.addHandler(Capture())
    policy = "alpha beta gamma delta"
    retrieve_baseline(policy, _config(4), _client(TextEncoder(None)), logger=logger)

    assert len(records) == 1
    fields = cast(dict[str, Any], getattr(records[0], "fields"))
    assert fields["model_version"] == REVISION
    assert fields["passage_count"] == 1
    assert fields["criterion_count"] == 22
    assert policy not in str(fields)


def test_cosine_rejects_a_length_mismatch() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1
    with pytest.raises(RetrievalError, match="same length"):
        cosine_similarity([1.0], [1.0, 0.0])

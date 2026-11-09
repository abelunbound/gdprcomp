import json
import logging
from typing import Any, cast

import pytest

from gdpr_gap.config import AppConfig, EmbeddingConfig, SegmentationConfig, load_config
from gdpr_gap.criteria import load_criteria
from gdpr_gap.embeddings import EmbeddingClient, HiddenStates
from gdpr_gap.llm import LLMClient, RawCompletion
from gdpr_gap.orchestrator import (
    DEFAULT_METHOD,
    OrchestratorError,
    PolicyDocument,
    run_batch,
    run_policy,
)
from gdpr_gap.segmentation import segment_passages

KEY = "test-key-not-a-real-secret"
NAME = "nlpaueb/legal-bert-base-uncased"
REVISION = "15b570cbf88259610b082a167dacc190124f60f6"


class YesTransport:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, model_id: str, prompt: str, timeout: float) -> RawCompletion:
        del model_id, timeout
        self.calls += 1
        marker = "criterion_id must be "
        start = prompt.index(marker) + len(marker)
        criterion_id = prompt[start:].split(".", maxsplit=1)[0]
        body = {
            "criterion_id": criterion_id,
            "verdict": "Yes",
            "justification": "The policy states the condition.",
            "evidence_quotes": [],
        }
        return RawCompletion(text=json.dumps(body), input_tokens=3, output_tokens=3)


class SplitEncoder:
    name = NAME
    version = REVISION

    def __init__(self, hit: str, passages: set[str]) -> None:
        self.hit = hit
        self.passages = passages
        self.calls = 0

    def encode(self, texts: list[str]) -> HiddenStates:
        self.calls += 1
        states: list[list[list[float]]] = []
        mask: list[list[int]] = []
        for text in texts:
            if text == self.hit:
                vector = [1.0, 0.0, 0.0]
            elif text in self.passages:
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]
            states.append([vector])
            mask.append([1])
        return HiddenStates(states, mask)


def _config(threshold: float | None = None) -> AppConfig:
    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )
    ordered = sorted(
        load_criteria(config.paths.criteria_path).criteria,
        key=lambda item: item.display_order,
    )
    words = len(ordered[0].condition.split())
    return config.model_copy(
        update={
            "segmentation": SegmentationConfig(passage_words=words, overlap_words=0),
            "thresholds": config.thresholds.model_copy(update={"retrieval_similarity": threshold}),
        }
    )


def _policy(config: AppConfig) -> tuple[str, SplitEncoder]:
    ordered = sorted(
        load_criteria(config.paths.criteria_path).criteria,
        key=lambda item: item.display_order,
    )
    condition = ordered[0].condition
    text = f"{condition} unrelated filler words extra"
    passages = {item.text for item in segment_passages(text, config.segmentation)}
    return text, SplitEncoder(condition, passages)


def _embeddings(encoder: SplitEncoder) -> EmbeddingClient:
    settings = EmbeddingConfig(model_name=NAME, revision=REVISION, batch_size=8)
    return EmbeddingClient(settings, encoder)


def test_default_method_is_whole_policy_and_is_recorded() -> None:
    config = _config()
    text, _encoder = _policy(config)
    transport = YesTransport()
    result = run_policy(text, config, policy_id="745", client=LLMClient(config, transport))

    assert DEFAULT_METHOD == "whole_policy_llm"
    assert result.method == "whole_policy_llm"
    assert result.policy_id == "745"
    assert result.review is not None
    assert result.hybrid is None
    assert result.retrieval is None
    assert len(result.review.judgement.verdicts) == 22
    assert transport.calls == 22


def test_batch_uses_the_same_method_on_every_result() -> None:
    config = _config()
    text, encoder = _policy(config)
    documents = [
        PolicyDocument(policy_id="745", text=text),
        PolicyDocument(policy_id="747", text=text),
    ]
    results = run_batch(documents, config, "legal_bert_only", embeddings=_embeddings(encoder))

    assert [item.method for item in results] == ["legal_bert_only", "legal_bert_only"]
    assert [item.policy_id for item in results] == ["745", "747"]
    assert all(item.retrieval is not None and item.review is None for item in results)
    assert encoder.calls > 0


def test_hybrid_records_its_method_and_unknown_method_makes_no_call() -> None:
    config = _config(0.5)
    text, encoder = _policy(config)
    transport = YesTransport()
    result = run_policy(
        text,
        config,
        "hybrid",
        policy_id="750",
        client=LLMClient(config, transport),
        embeddings=_embeddings(encoder),
    )

    assert result.method == "hybrid"
    assert result.hybrid is not None
    assert result.hybrid.method == "hybrid"
    assert result.review is None
    assert transport.calls == 1

    before = transport.calls
    with pytest.raises(OrchestratorError, match="Unknown method"):
        run_policy(text, config, "bert", policy_id="750", client=LLMClient(config, transport))
    assert transport.calls == before


def test_missing_client_and_empty_batch_are_rejected() -> None:
    config = _config()
    with pytest.raises(OrchestratorError, match="LLM client"):
        run_policy("a policy", config, policy_id="745")
    with pytest.raises(OrchestratorError, match="no policies"):
        run_batch([], config, client=LLMClient(config, YesTransport()))
    with pytest.raises(OrchestratorError, match="unique"):
        run_batch(
            [
                PolicyDocument(policy_id="745", text="one"),
                PolicyDocument(policy_id="745", text="two"),
            ],
            config,
            client=LLMClient(config, YesTransport()),
        )


def test_run_log_records_method_and_policy_id_only() -> None:
    config = _config()
    text, encoder = _policy(config)
    logger = logging.getLogger("gdpr_gap_orchestrator_test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger.addHandler(Capture())
    run_policy(
        text,
        config,
        "legal_bert_only",
        policy_id="745",
        embeddings=_embeddings(encoder),
        logger=logger,
    )
    runs = [record for record in records if getattr(record, "event", "") == "policy_run"]
    fields = cast(dict[str, Any], getattr(runs[0], "fields"))
    assert fields["method"] == "legal_bert_only"
    assert fields["policy_id"] == "745"
    assert text not in str(fields)

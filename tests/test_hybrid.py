import json
import logging
from typing import Any, cast

import pytest

from gdpr_gap.config import AppConfig, EmbeddingConfig, SegmentationConfig, load_config
from gdpr_gap.criteria import load_criteria
from gdpr_gap.embeddings import EmbeddingClient, HiddenStates
from gdpr_gap.hybrid import HybridError, judge_hybrid
from gdpr_gap.judge import judge_policy
from gdpr_gap.llm import LLMClient, RawCompletion
from gdpr_gap.segmentation import segment_passages

KEY = "test-key-not-a-real-secret"
NAME = "nlpaueb/legal-bert-base-uncased"
REVISION = "15b570cbf88259610b082a167dacc190124f60f6"


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


class RecordingTransport:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, *, model_id: str, prompt: str, timeout: float) -> RawCompletion:
        self.prompts.append(prompt)
        criterion_id = _criterion_id(prompt)
        body = {
            "criterion_id": criterion_id,
            "verdict": "Yes",
            "justification": "Judged from the retrieved passages.",
            "evidence_quotes": [],
        }
        return RawCompletion(text=json.dumps(body), input_tokens=5, output_tokens=5)


def _config(passage_words: int, threshold: float | None) -> AppConfig:
    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )
    return config.model_copy(
        update={
            "segmentation": SegmentationConfig(passage_words=passage_words, overlap_words=0),
            "thresholds": config.thresholds.model_copy(update={"retrieval_similarity": threshold}),
        }
    )


def _embeddings(encoder: SplitEncoder) -> EmbeddingClient:
    settings = EmbeddingConfig(model_name=NAME, revision=REVISION, batch_size=8)
    return EmbeddingClient(settings, encoder)


def _criterion_id(prompt: str) -> str:
    marker = "criterion_id must be "
    start = prompt.index(marker) + len(marker)
    return prompt[start:].split(".", maxsplit=1)[0]


def _policy() -> tuple[AppConfig, str, SplitEncoder, list[str]]:
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
    tuned = _config(len(condition.split()), 0.5)
    policy = f"{condition} unrelated filler words extra"
    passages = segment_passages(policy, tuned.segmentation)
    encoder = SplitEncoder(condition, {item.text for item in passages})
    return tuned, policy, encoder, [item.id for item in ordered]


def test_relevant_passage_is_judged_and_misses_skip_the_model() -> None:
    config, policy, encoder, criterion_ids = _policy()
    transport = RecordingTransport()
    logger = logging.getLogger("gdpr_gap_hybrid_test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger.addHandler(Capture())
    hybrid = judge_hybrid(
        policy,
        config,
        _embeddings(encoder),
        LLMClient(config, transport),
        logger=logger,
    )
    whole = judge_policy(policy, config, LLMClient(config, RecordingTransport()))

    assert hybrid.method == "hybrid"
    assert hybrid.threshold == 0.5
    assert hybrid.embedding_model_version == REVISION
    assert [item.judgement.criterion_id for item in hybrid.judgement.verdicts] == criterion_ids
    assert [item.judgement.criterion_id for item in whole.verdicts] == criterion_ids
    assert len(transport.prompts) == 1
    assert len(whole.verdicts) == 22
    hit = criterion_ids[0]
    assert hybrid.retrieval_misses == criterion_ids[1:]
    assert hybrid.judgement.verdicts[0].judgement.verdict == "Yes"
    assert hybrid.judgement.verdicts[1].judgement.verdict == "No"
    assert hybrid.judgement.verdicts[1].usage.input_tokens == 0
    passages = segment_passages(policy, config.segmentation)
    assert passages[0].text in transport.prompts[0]
    assert passages[1].text not in transport.prompts[0]
    misses = [record for record in records if record.getMessage() == "retrieval_miss"]
    miss_ids = {
        cast(dict[str, Any], getattr(record, "fields"))["criterion_id"] for record in misses
    }
    assert len(misses) == 21
    assert miss_ids == set(criterion_ids[1:])
    assert all(policy not in str(getattr(record, "fields")) for record in misses)
    summary = next(record for record in records if record.getMessage() == "hybrid_judged")
    summary_fields = cast(dict[str, Any], getattr(summary, "fields"))
    assert summary_fields["retrieval_miss_count"] == 21
    assert summary_fields["model_call_count"] == 1
    assert hit not in hybrid.retrieval_misses


def test_missing_threshold_makes_no_model_or_embedding_call() -> None:
    config, policy, encoder, _ids = _policy()
    bare = config.model_copy(
        update={"thresholds": config.thresholds.model_copy(update={"retrieval_similarity": None})}
    )
    transport = RecordingTransport()

    with pytest.raises(HybridError, match="calibrated"):
        judge_hybrid(policy, bare, _embeddings(encoder), LLMClient(bare, transport))

    assert transport.prompts == []
    assert encoder.calls == 0

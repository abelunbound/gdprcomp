import json
from pathlib import Path

import pytest

from gdpr_gap.config import AppConfig, EmbeddingConfig, load_config
from gdpr_gap.embeddings import EmbeddingClient, HiddenStates
from gdpr_gap.llm import LLMClient, RawCompletion
from gdpr_gap.orchestrator import PolicyDocument
from gdpr_gap.repeats import RepeatError, agreement_report, load_repeats, record_repeats

KEY = "test-key-not-a-real-secret"
REPEAT_ID = "22222222-2222-4222-8222-222222222222"
POLICY = "We keep records of processing activities."
NAME = "nlpaueb/legal-bert-base-uncased"
REVISION = "15b570cbf88259610b082a167dacc190124f60f6"


class FlipTransport:
    """Second repeat marks the first criterion No. Every other verdict is Yes."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, model_id: str, prompt: str, timeout: float) -> RawCompletion:
        del model_id, timeout
        self.calls += 1
        index = (self.calls - 1) % 22
        repeat = (self.calls - 1) // 22
        verdict = "No" if index == 0 and repeat == 1 else "Yes"
        body = {
            "criterion_id": _criterion_id(prompt),
            "verdict": verdict,
            "justification": "The policy states the condition.",
            "evidence_quotes": [],
        }
        return RawCompletion(text=json.dumps(body), input_tokens=2, output_tokens=2)


class FixedEncoder:
    name = NAME
    version = REVISION

    def encode(self, texts: list[str]) -> HiddenStates:
        return HiddenStates([[[1.0, 0.0]] for _text in texts], [[1] for _text in texts])


def _criterion_id(prompt: str) -> str:
    marker = "criterion_id must be "
    start = prompt.index(marker) + len(marker)
    return prompt[start:].split(".", maxsplit=1)[0]


def _config() -> AppConfig:
    return load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )


def test_repeat_agreement_is_stored_per_criterion_and_overall(tmp_path: Path) -> None:
    config = _config()
    document = PolicyDocument(policy_id="745", text=POLICY)
    report = record_repeats(
        [document],
        config,
        repeats=2,
        client=LLMClient(config, FlipTransport()),
        feedback=False,
        directory=tmp_path,
        repeat_id=REPEAT_ID,
    )
    path = tmp_path / f"{REPEAT_ID}.json"
    loaded = load_repeats(path)
    text = agreement_report(loaded)
    changed = loaded.criteria[0]
    stable = loaded.criteria[1]

    assert loaded.model_dump(mode="json") == report.model_dump(mode="json")
    assert loaded.method == "whole_policy_llm"
    assert loaded.repeats == 2
    assert changed.agreement == 0
    assert changed.unanimous is False
    assert changed.by_policy[0].labels == ["Yes", "No"]
    assert stable.agreement == 1
    assert stable.unanimous is True
    assert loaded.overall_agreeing_pairs == 21
    assert loaded.overall_pairs == 22
    assert loaded.overall_agreement == pytest.approx(21 / 22)
    assert f"Overall agreement: 21/22 ({21 / 22:.3f})" in text
    assert f"{changed.criterion_id}: 0/1 (0.000)" in text
    assert f"{stable.criterion_id}: 1/1 (1.000)" in text
    assert KEY not in path.read_text(encoding="utf-8")

    with pytest.raises(RepeatError, match="overwrite"):
        record_repeats(
            [document],
            config,
            repeats=2,
            client=LLMClient(config, FlipTransport()),
            feedback=False,
            directory=tmp_path,
            repeat_id=REPEAT_ID,
        )


def test_retrieval_repeats_agree_and_too_few_repeats_write_nothing(tmp_path: Path) -> None:
    config = _config()
    settings = EmbeddingConfig(model_name=NAME, revision=REVISION, batch_size=8)
    document = PolicyDocument(policy_id="745", text=POLICY)
    report = record_repeats(
        [document],
        config,
        "legal_bert_only",
        repeats=2,
        embeddings=EmbeddingClient(settings, FixedEncoder()),
        directory=tmp_path,
        repeat_id=REPEAT_ID,
    )

    assert report.method == "legal_bert_only"
    assert report.overall_agreement == 1
    assert all(item.unanimous for item in report.criteria)
    assert "Overall agreement: 22/22 (1.000)" in agreement_report(report)

    with pytest.raises(RepeatError, match="2 to 20"):
        record_repeats(
            [document],
            config,
            repeats=1,
            client=LLMClient(config, FlipTransport()),
            directory=tmp_path,
            repeat_id="33333333-3333-4333-8333-333333333333",
        )
    assert [path.name for path in tmp_path.iterdir()] == [f"{REPEAT_ID}.json"]

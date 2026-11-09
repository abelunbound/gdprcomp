import json
import logging
from pathlib import Path
from typing import Any, cast

import pytest

from gdpr_gap.config import AppConfig, load_config
from gdpr_gap.criteria import load_criteria
from gdpr_gap.judge import JudgeError, judge_policy
from gdpr_gap.llm import LLMClient, RawCompletion
from gdpr_gap.result_cache import ResultCache

KEY = "test-key-not-a-real-secret"


def _config() -> AppConfig:
    return load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )


class RecordingTransport:
    def __init__(self, verdict: str = "Yes", criterion_id: str | None = None) -> None:
        self.prompts: list[str] = []
        self.verdict = verdict
        self.criterion_id = criterion_id

    def complete(self, *, model_id: str, prompt: str, timeout: float) -> RawCompletion:
        self.prompts.append(prompt)
        criterion_id = self.criterion_id or _criterion_id(prompt)
        body = {
            "criterion_id": criterion_id,
            "verdict": self.verdict,
            "justification": "Judged from the full policy.",
            "evidence_quotes": ["w2500"],
        }
        return RawCompletion(text=json.dumps(body), input_tokens=5, output_tokens=5)


def _criterion_id(prompt: str) -> str:
    marker = "criterion_id must be "
    start = prompt.index(marker) + len(marker)
    return prompt[start:].split(".", maxsplit=1)[0]


def _policy(words: int) -> str:
    return " ".join(f"w{index}" for index in range(words))


def test_long_policy_yields_22_verdicts_from_the_full_text() -> None:
    config = _config()
    policy = _policy(2501)
    transport = RecordingTransport()
    client = LLMClient(config, transport)

    judgement = judge_policy(policy, config, client)

    catalog = load_criteria(config.paths.criteria_path)
    ordered = [item.id for item in sorted(catalog.criteria, key=lambda item: item.display_order)]
    assert len(policy.split()) > 2500
    assert [item.judgement.criterion_id for item in judgement.verdicts] == ordered
    assert {item.judgement.verdict for item in judgement.verdicts} == {"Yes"}
    assert len(transport.prompts) == 22
    assert all(policy in prompt for prompt in transport.prompts)
    conditions = {item.condition for item in catalog.criteria}
    assert all(any(condition in prompt for prompt in transport.prompts) for condition in conditions)
    assert judgement.prompt_id == "whole_policy_judge"
    assert judgement.prompt_version == "v1"


def test_over_limit_policy_is_not_sent() -> None:
    config = _config()
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"max_input_tokens": 10})}
    )
    policy = " ".join(f"w{index}" for index in range(400))
    transport = RecordingTransport()
    logger = logging.getLogger("gdpr_gap_judge_limit_test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger.addHandler(Capture())

    with pytest.raises(JudgeError, match="was not sent") as exc_info:
        judge_policy(policy, config, LLMClient(config, transport), logger=logger)

    message = str(exc_info.value)
    assert "w399" not in message
    assert transport.prompts == []
    assert records[0].getMessage() == "policy_over_limit"
    fields = cast(dict[str, Any], getattr(records[0], "fields"))
    assert fields["word_count"] == 400
    assert fields["max_input_tokens"] == 10
    assert fields["input_tokens"] > 10
    assert policy not in json.dumps(fields)


def test_empty_policy_makes_no_model_call() -> None:
    config = _config()
    transport = RecordingTransport()

    with pytest.raises(JudgeError, match="empty"):
        judge_policy("  \n", config, LLMClient(config, transport))

    assert transport.prompts == []


def test_wrong_criterion_id_is_rejected() -> None:
    config = _config()
    transport = RecordingTransport(criterion_id="not_a_criterion")

    with pytest.raises(JudgeError, match="not_a_criterion"):
        judge_policy("A short policy.", config, LLMClient(config, transport))

    assert len(transport.prompts) == 1


def test_cached_policy_is_not_judged_twice(tmp_path: Path) -> None:
    config = _config()
    transport = RecordingTransport()
    client = LLMClient(config, transport)
    cache = ResultCache(tmp_path)
    policy = "We ask for consent before sending marketing email."

    first = judge_policy(policy, config, client, cache=cache)
    second = judge_policy(policy, config, client, cache=cache)

    assert first == second
    assert len(transport.prompts) == 22

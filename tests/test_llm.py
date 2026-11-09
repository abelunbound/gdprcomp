import json
import logging
import urllib.error
import urllib.request
from email.message import Message

import pytest

from gdpr_gap.config import AppConfig, load_config
from gdpr_gap.llm import (
    AnthropicTransport,
    CriterionJudgement,
    LLMClient,
    ModelError,
    ModelResult,
    RawCompletion,
    RetryableModelError,
    build_transport,
    parse_judgement,
)

VALID = {
    "criterion_id": "consent_mechanisms",
    "verdict": "Yes",
    "justification": "The sentence says consent is requested.",
    "evidence_quotes": ["We ask for consent before sending marketing email."],
}


def _config() -> AppConfig:
    return load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": "test-key-not-a-real-secret"},
        load_env_file=False,
    )


class FakeTransport:
    def __init__(self, results: list[RawCompletion | Exception]) -> None:
        self._results = list(results)
        self.calls = 0

    def complete(self, *, model_id: str, prompt: str, timeout: float) -> RawCompletion:
        self.calls += 1
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _judge(client: LLMClient, prompt: str = "prompt") -> ModelResult:
    return client.judge(prompt, prompt_id="whole_policy_judge", prompt_version="v1")


def _raw(text: str) -> RawCompletion:
    return RawCompletion(text=text, input_tokens=12, output_tokens=8)


def test_valid_json_is_parsed_and_token_use_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("gdpr_gap_test_llm")
    logger.setLevel(logging.INFO)
    transport = FakeTransport([_raw(json.dumps(VALID))])
    client = LLMClient(
        _config(),
        transport,
        logger=logger,
        input_cost_per_million=1,
        output_cost_per_million=5,
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = _judge(client)

    assert result.judgement == CriterionJudgement.model_validate(VALID)
    assert result.prompt_id == "whole_policy_judge"
    assert result.prompt_version == "v1"
    assert result.usage.input_tokens == 12
    assert result.usage.cost_usd == pytest.approx((12 * 1 + 8 * 5) / 1_000_000)
    events = [record.getMessage() for record in caplog.records]
    assert "model_call" in events


def test_fenced_json_is_accepted_and_bad_verdict_is_not_retried() -> None:
    fenced = "```json\n" + json.dumps(VALID) + "\n```"
    assert parse_judgement(fenced).verdict == "Yes"

    bad = dict(VALID)
    bad["verdict"] = "yes"
    transport = FakeTransport([_raw(json.dumps(bad)), _raw(json.dumps(VALID))])
    client = LLMClient(_config(), transport, sleep=lambda _seconds: None)

    with pytest.raises(ModelError, match="schema"):
        _judge(client)
    assert transport.calls == 1


def test_temporary_provider_errors_are_retried() -> None:
    transport = FakeTransport([RetryableModelError("Anthropic HTTP 529."), _raw(json.dumps(VALID))])
    config = _config()
    config = config.model_copy(update={"model": config.model.model_copy(update={"max_retries": 2})})
    client = LLMClient(config, transport, sleep=lambda _seconds: None)

    result = _judge(client)

    assert result.judgement.criterion_id == "consent_mechanisms"
    assert transport.calls == 2


def test_auth_failure_is_not_retried() -> None:
    transport = FakeTransport([ModelError("Anthropic HTTP 401."), _raw(json.dumps(VALID))])
    client = LLMClient(_config(), transport, sleep=lambda _seconds: None)

    with pytest.raises(ModelError, match="401"):
        _judge(client)
    assert transport.calls == 1


def test_anthropic_transport_reads_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "content": [
            {"type": "thinking", "thinking": "hidden"},
            {"type": "text", "text": json.dumps(VALID)},
        ],
        "usage": {"input_tokens": 4, "output_tokens": 6},
    }

    class _Response:
        def read(self) -> bytes:
            return json.dumps(body).encode("utf-8")

        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def urlopen(
        request: urllib.request.Request, timeout: float, context: object = None
    ) -> _Response:
        assert timeout == 5
        assert request.get_header("X-api-key") == "test-key-not-a-real-secret"
        assert request.get_header("Anthropic-version") == "2023-06-01"
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    raw = AnthropicTransport("test-key-not-a-real-secret", 100).complete(
        model_id="claude-haiku-5-5", prompt="prompt", timeout=5
    )

    assert json.loads(raw.text)["verdict"] == "Yes"
    assert raw.input_tokens == 4
    assert raw.output_tokens == 6


def test_http_500_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(request: urllib.request.Request, timeout: float, context: object = None) -> object:
        raise urllib.error.HTTPError(request.full_url, 500, "error", hdrs=Message(), fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    with pytest.raises(RetryableModelError, match="HTTP 500"):
        AnthropicTransport("test-key", 100).complete(model_id="m", prompt="p", timeout=1)


def test_build_transport_uses_the_configured_provider() -> None:
    assert isinstance(build_transport(_config()), AnthropicTransport)


def test_cost_is_absent_when_no_price_is_configured() -> None:
    client = LLMClient(_config(), FakeTransport([_raw(json.dumps(VALID))]))
    assert _judge(client).usage.cost_usd is None

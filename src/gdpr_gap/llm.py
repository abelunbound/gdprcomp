"""Provider-agnostic model client with a strict criterion-judgement schema.

Retries cover timeouts and temporary provider errors. A response that is not
the schema `{criterion_id, verdict, justification, evidence_quotes}` fails
once and is logged. Token counts are logged. Cost is logged only when a price
is supplied, because this client does not invent a price.
"""

from __future__ import annotations

import json
import logging
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gdpr_gap.config import AppConfig
from gdpr_gap.logging import log_event

Verdict = Literal["Yes", "Somewhat", "No"]


class ModelError(Exception):
    """Raised when a model call cannot return a valid judgement."""


class RetryableModelError(ModelError):
    """A timeout or temporary provider error. The client may try again."""


class CriterionJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    verdict: Verdict
    justification: str = Field(min_length=1)
    evidence_quotes: list[str]


class TokenUse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float | None = None


class ModelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgement: CriterionJudgement
    model_id: str
    usage: TokenUse


class RawCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ChatTransport(Protocol):
    def complete(self, *, model_id: str, prompt: str, timeout: float) -> RawCompletion:
        """Send one prompt and return the raw model text plus token counts."""


class LLMClient:
    """Call the configured model and parse one criterion judgement."""

    def __init__(
        self,
        config: AppConfig,
        transport: ChatTransport,
        *,
        logger: logging.Logger | None = None,
        sleep: Callable[[float], None] = time.sleep,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._logger = logger
        self._sleep = sleep
        self._input_cost = input_cost_per_million
        self._output_cost = output_cost_per_million

    def judge(self, prompt: str, *, model_id: str | None = None) -> ModelResult:
        """Return one schema-valid judgement. Malformed text is not retried."""
        chosen_model = model_id or self._config.model.id
        attempts = self._config.model.max_retries
        timeout = self._config.model.timeout_seconds
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                raw = self._transport.complete(
                    model_id=chosen_model, prompt=prompt, timeout=timeout
                )
            except ModelError as exc:
                last_error = exc
                self._log(
                    "model_call_failed",
                    model_id=chosen_model,
                    attempt=attempt,
                    reason=str(exc),
                )
                if not isinstance(exc, RetryableModelError) or attempt == attempts:
                    break
                self._sleep(0.5 * attempt)
                continue
            try:
                judgement = parse_judgement(raw.text)
            except ModelError as exc:
                self._log(
                    "model_response_invalid",
                    model_id=chosen_model,
                    attempt=attempt,
                    reason=str(exc),
                )
                raise
            usage = TokenUse(
                input_tokens=raw.input_tokens,
                output_tokens=raw.output_tokens,
                cost_usd=_cost(raw, self._input_cost, self._output_cost),
            )
            self._log(
                "model_call",
                model_id=chosen_model,
                attempt=attempt,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
            )
            return ModelResult(judgement=judgement, model_id=chosen_model, usage=usage)
        raise ModelError(f"Model call failed after {attempts} attempts: {last_error}")

    def _log(self, event: str, **fields: str | int | float | bool | None) -> None:
        if self._logger is not None:
            log_event(self._logger, event, **fields)


def parse_judgement(text: str) -> CriterionJudgement:
    """Parse a model reply. A fence around one JSON object is removed; other text fails."""
    payload = _json_object(text)
    if not isinstance(payload, dict):
        raise ModelError("Model response must be a JSON object.")
    try:
        return CriterionJudgement.model_validate(payload)
    except ValidationError as exc:
        raise ModelError(f"Model response does not match the judgement schema: {exc}") from exc


def build_transport(config: AppConfig) -> ChatTransport:
    """Build the HTTP transport for the configured provider."""
    key = config.api_key.get_secret_value()
    tokens = config.model.max_output_tokens
    if config.model.provider == "claude":
        return AnthropicTransport(key, tokens)
    return OpenAITransport(key, tokens)


class AnthropicTransport:
    def __init__(self, api_key: str, max_output_tokens: int) -> None:
        self._api_key = api_key
        self._max_output_tokens = max_output_tokens

    def complete(self, *, model_id: str, prompt: str, timeout: float) -> RawCompletion:
        body = json.dumps(
            {
                "model": model_id,
                "max_tokens": self._max_output_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        payload = _read_http(request, timeout, "Anthropic")
        content = payload.get("content")
        usage = payload.get("usage")
        if not isinstance(content, list) or not content or not isinstance(usage, dict):
            raise ModelError("Anthropic response is missing content or usage.")
        text = _anthropic_text(content)
        if text is None:
            raise ModelError("Anthropic response content has no text.")
        return RawCompletion(
            text=text,
            input_tokens=_token_count(usage.get("input_tokens")),
            output_tokens=_token_count(usage.get("output_tokens")),
        )


class OpenAITransport:
    def __init__(self, api_key: str, max_output_tokens: int) -> None:
        self._api_key = api_key
        self._max_output_tokens = max_output_tokens

    def complete(self, *, model_id: str, prompt: str, timeout: float) -> RawCompletion:
        body = json.dumps(
            {
                "model": model_id,
                "max_tokens": self._max_output_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        payload = _read_http(request, timeout, "OpenAI")
        choices = payload.get("choices")
        usage = payload.get("usage")
        if not isinstance(choices, list) or not choices or not isinstance(usage, dict):
            raise ModelError("OpenAI response is missing choices or usage.")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ModelError("OpenAI response message has no text.")
        return RawCompletion(
            text=content,
            input_tokens=_token_count(usage.get("prompt_tokens")),
            output_tokens=_token_count(usage.get("completion_tokens")),
        )


def _read_http(request: urllib.request.Request, timeout: float, provider: str) -> dict[str, object]:
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        message = f"{provider} HTTP {exc.code}."
        if exc.code == 429 or exc.code >= 500:
            raise RetryableModelError(message) from exc
        raise ModelError(message) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RetryableModelError(f"{provider} request failed: {exc}") from exc
    try:
        loaded = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelError(f"{provider} returned a non-JSON body.") from exc
    if not isinstance(loaded, dict):
        raise ModelError(f"{provider} returned a non-object body.")
    return cast(dict[str, object], loaded)


def _ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    if context.cert_store_stats().get("x509_ca", 0) > 0:
        return context
    for path in (Path("/etc/ssl/cert.pem"), Path("/private/etc/ssl/cert.pem")):
        if path.is_file():
            context.load_verify_locations(path)
            return context
    return context


def _json_object(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ModelError("Model response is not JSON.") from exc


def _anthropic_text(content: list[object]) -> str | None:
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            return text
    return None


def _token_count(value: object) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    raise ModelError("Model response has an invalid token count.")


def _cost(
    raw: RawCompletion, input_price: float | None, output_price: float | None
) -> float | None:
    if input_price is None or output_price is None:
        return None
    return (raw.input_tokens * input_price + raw.output_tokens * output_price) / 1_000_000

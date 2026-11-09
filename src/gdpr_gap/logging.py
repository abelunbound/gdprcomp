"""Structured logs that redact secrets and omit policy text by default."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping
from typing import TextIO

from gdpr_gap.config import AppConfig

REDACTED = "[REDACTED]"
_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{20,}")
_SETTINGS: dict[str, LogSettings] = {}
FieldValue = str | int | float | bool | None


class LogSettings:
    """Secrets and the research flag for one configured logger."""

    def __init__(self, secrets: list[str], *, log_policy_text: bool) -> None:
        self.log_policy_text = log_policy_text
        long_enough = {secret for secret in secrets if len(secret) >= 8}
        self._secrets = sorted(long_enough, key=len, reverse=True)

    def add_secret(self, value: str) -> None:
        if len(value) < 8 or value in self._secrets:
            return
        self._secrets.append(value)
        self._secrets.sort(key=len, reverse=True)

    def redact(self, text: str) -> str:
        redacted = text
        for secret in self._secrets:
            redacted = redacted.replace(secret, REDACTED)
        return _KEY_PATTERN.sub(REDACTED, redacted)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def __init__(self, settings: LogSettings) -> None:
        super().__init__()
        self._settings = settings

    def format(self, record: logging.LogRecord) -> str:
        event = str(getattr(record, "event", record.getMessage()))
        raw_fields = getattr(record, "fields", {})
        fields = raw_fields if isinstance(raw_fields, Mapping) else {}
        payload: dict[str, object] = {
            "event": self._settings.redact(event),
            "level": record.levelname,
            "logger": record.name,
        }
        for key, value in fields.items():
            if key == "policy_text" and not self._settings.log_policy_text:
                continue
            payload[str(key)] = _json_value(value, self._settings)
        return json.dumps(payload, sort_keys=True)


def configure_logging(config: AppConfig, *, stream: TextIO | None = None) -> logging.Logger:
    """Attach a JSON handler to the package logger."""
    logger = logging.getLogger("gdpr_gap")
    settings = LogSettings(
        [config.api_key.get_secret_value()],
        log_policy_text=config.log_policy_text,
    )
    _SETTINGS[logger.name] = settings
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter(settings))
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: FieldValue) -> None:
    """Write one structured event. String fields are redacted."""
    logger.info(event, extra={"event": event, "fields": fields})


def log_policy(logger: logging.Logger, policy_text: str, **fields: FieldValue) -> None:
    """Log policy text only when the research flag is on."""
    settings = _require_settings(logger)
    if settings.log_policy_text:
        log_event(logger, "policy_text", policy_text=policy_text, **fields)
        return
    settings.add_secret(policy_text)
    log_event(logger, "policy_text_omitted", policy_chars=len(policy_text), **fields)


def _require_settings(logger: logging.Logger) -> LogSettings:
    try:
        return _SETTINGS[logger.name]
    except KeyError as exc:
        raise RuntimeError("Call configure_logging before writing log events.") from exc


def _json_value(value: object, settings: LogSettings) -> object:
    if isinstance(value, str):
        return settings.redact(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return settings.redact(str(value))

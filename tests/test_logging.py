import logging
from io import StringIO

from pydantic import SecretStr

from gdpr_gap.config import load_config
from gdpr_gap.logging import configure_logging, log_event, log_policy

SECRET = "sk-ant-testkeyvalue1234567890"
POLICY = "Unique policy sentence about personal data retention for this test."


def _logger(stream: StringIO, *, log_policy_text: bool = False) -> logging.Logger:
    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": SECRET},
        load_env_file=False,
    )
    config = config.model_copy(
        update={"log_policy_text": log_policy_text, "api_key": SecretStr(SECRET)}
    )
    return configure_logging(config, stream=stream)


def test_secrets_are_redacted_in_messages_and_fields() -> None:
    stream = StringIO()
    logger = _logger(stream)
    log_event(logger, f"provider rejected {SECRET}", detail=f"header {SECRET}")
    output = stream.getvalue()
    assert SECRET not in output
    assert "[REDACTED]" in output


def test_unregistered_api_key_pattern_is_redacted() -> None:
    stream = StringIO()
    logger = _logger(stream)
    leaked = "sk-ant-anotherkeyvalue123456"
    log_event(logger, f"saw {leaked}")
    assert leaked not in stream.getvalue()
    assert "[REDACTED]" in stream.getvalue()


def test_policy_text_is_omitted_unless_the_research_flag_is_on() -> None:
    hidden = StringIO()
    logger = _logger(hidden)
    log_policy(logger, POLICY)
    log_event(logger, "later", note=POLICY)
    hidden_output = hidden.getvalue()
    assert POLICY not in hidden_output
    assert "policy_text_omitted" in hidden_output

    shown = StringIO()
    log_policy(_logger(shown, log_policy_text=True), POLICY)
    shown_output = shown.getvalue()
    assert POLICY in shown_output
    assert "policy_text" in shown_output


def test_research_logs_still_redact_secrets() -> None:
    stream = StringIO()
    logger = _logger(stream, log_policy_text=True)
    log_policy(logger, f"{POLICY} Contact key {SECRET}.")
    output = stream.getvalue()
    assert SECRET not in output
    assert POLICY in output
    assert "[REDACTED]" in output

"""GDPR compliance gap evaluator."""

from gdpr_gap.config import ConfigError, MissingSecretError, load_config
from gdpr_gap.logging import configure_logging, log_event, log_policy

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "MissingSecretError",
    "configure_logging",
    "load_config",
    "log_event",
    "log_policy",
    "ping",
]


def ping() -> str:
    """Return a marker so the installed package can be import-checked."""
    return "ok"

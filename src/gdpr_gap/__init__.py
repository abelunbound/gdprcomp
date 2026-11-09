"""GDPR compliance gap evaluator."""

from gdpr_gap.config import ConfigError, MissingSecretError, load_config

__version__ = "0.1.0"

__all__ = ["ConfigError", "MissingSecretError", "load_config", "ping"]


def ping() -> str:
    """Return a marker so the installed package can be import-checked."""
    return "ok"

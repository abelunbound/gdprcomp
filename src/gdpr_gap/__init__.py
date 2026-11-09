"""GDPR compliance gap evaluator."""

from gdpr_gap.config import ConfigError, MissingSecretError, load_config
from gdpr_gap.criteria import CriteriaError, load_criteria, summary_table
from gdpr_gap.gaps import gap_ids, is_gap
from gdpr_gap.logging import configure_logging, log_event, log_policy
from gdpr_gap.scoring import ScoringError, score_policy

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "CriteriaError",
    "MissingSecretError",
    "configure_logging",
    "gap_ids",
    "is_gap",
    "load_config",
    "load_criteria",
    "log_event",
    "log_policy",
    "ping",
    "score_policy",
    "ScoringError",
    "summary_table",
]


def ping() -> str:
    """Return a marker so the installed package can be import-checked."""
    return "ok"

"""GDPR compliance gap evaluator."""

from gdpr_gap.config import ConfigError, MissingSecretError, load_config
from gdpr_gap.corpus import CorpusError, load_corpus
from gdpr_gap.criteria import CriteriaError, load_criteria, summary_table
from gdpr_gap.embeddings import EmbeddingClient, EmbeddingError
from gdpr_gap.exports import ExportError, export_classification, export_feedback_quality
from gdpr_gap.extract import ExtractionError, extract_text, extract_url
from gdpr_gap.feedback import CitationReport, check_citations, evaluate_policy, suggest_fixes
from gdpr_gap.gaps import gap_ids, is_gap
from gdpr_gap.judge import JudgeError, judge_policy
from gdpr_gap.llm import LLMClient, ModelError, build_transport
from gdpr_gap.logging import configure_logging, log_event, log_policy
from gdpr_gap.prompts import PromptError, PromptRecord, PromptRegistry
from gdpr_gap.result_cache import CacheError, CacheKey, ResultCache, make_cache_key
from gdpr_gap.sampling import (
    SamplingError,
    load_gold_manifest,
    sample_descriptive_set,
    sample_gold_set,
)
from gdpr_gap.scoring import ScoringError, score_policy
from gdpr_gap.segmentation import SegmentationError, segment_passages

__version__ = "0.1.0"

__all__ = [
    "CacheError",
    "CitationReport",
    "check_citations",
    "CacheKey",
    "ConfigError",
    "CorpusError",
    "CriteriaError",
    "MissingSecretError",
    "configure_logging",
    "EmbeddingClient",
    "EmbeddingError",
    "extract_text",
    "extract_url",
    "evaluate_policy",
    "export_classification",
    "export_feedback_quality",
    "ExportError",
    "ExtractionError",
    "gap_ids",
    "is_gap",
    "JudgeError",
    "judge_policy",
    "load_config",
    "load_corpus",
    "load_criteria",
    "load_gold_manifest",
    "make_cache_key",
    "log_event",
    "LLMClient",
    "ModelError",
    "build_transport",
    "log_policy",
    "ping",
    "PromptError",
    "PromptRecord",
    "PromptRegistry",
    "ResultCache",
    "sample_descriptive_set",
    "sample_gold_set",
    "segment_passages",
    "score_policy",
    "suggest_fixes",
    "SegmentationError",
    "SamplingError",
    "ScoringError",
    "summary_table",
]


def ping() -> str:
    """Return a marker so the installed package can be import-checked."""
    return "ok"

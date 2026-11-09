"""Persist one orchestrated run so it can be reloaded without the original objects.

The file records the method, seed, model, prompts, library versions, corpus
texts, and every per-policy result. It does not record the API key. Loading
the file rebuilds the recorded results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gdpr_gap.calibration import load_calibration
from gdpr_gap.calibration import versions as calibration_versions
from gdpr_gap.config import AppConfig
from gdpr_gap.embeddings import EmbeddingClient
from gdpr_gap.llm import LLMClient, TokenUse
from gdpr_gap.logging import log_event
from gdpr_gap.orchestrator import (
    DEFAULT_METHOD,
    Method,
    PolicyDocument,
    PolicyRun,
    run_policy,
)
from gdpr_gap.prompts import PromptRegistry
from gdpr_gap.result_cache import ResultCache

_REQUIRED_LIBRARIES = ("gdpr-gap", "pydantic", "PyYAML", "python-dotenv", "dash")
_OPTIONAL_LIBRARIES = ("numpy", "torch", "transformers")
_RUN_ID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


class ManifestError(Exception):
    """Raised when a run manifest cannot be written or loaded."""


class LibraryVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class PromptPin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(pattern=r"^[a-z0-9_]+$")
    version: str = Field(pattern=r"^v[1-9][0-9]*$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1)


class CorpusItem(BaseModel):
    """One policy text that was scored. The hash identifies that exact text."""

    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    char_count: int = Field(ge=0)
    text: str


class CalibrationPin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(pattern=r"^v[1-9][0-9]*$")
    threshold: float = Field(ge=0, le=1)
    label_source: Literal["model-produced"]


class StoredPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    runtime_seconds: float = Field(ge=0)
    cost_usd: float | None
    result: PolicyRun


class RunSettings(BaseModel):
    """Non-secret settings required to repeat the run. The API key is absent."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    truncation: Literal["error"]
    method: Method
    provider: Literal["claude", "openai"]
    model_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    gold_id: str = Field(min_length=1)
    max_output_tokens: int = Field(ge=1)
    embedding_model_name: str = Field(min_length=1)
    embedding_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    passage_words: int = Field(ge=1)
    overlap_words: int = Field(ge=0)
    retrieval_similarity: float | None = Field(default=None, ge=0, le=1)
    corpus_location: str = Field(min_length=1)
    weight_mode: Literal["equal"]
    weight_value: float = Field(gt=0)
    yes_score: float
    somewhat_score: float
    no_score: float
    high_band_min: float
    mid_band_min: float


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(pattern=_RUN_ID)
    settings: RunSettings
    prompts: list[PromptPin] = Field(min_length=1)
    libraries: list[LibraryVersion] = Field(min_length=1)
    python_version: str = Field(min_length=1)
    calibration: CalibrationPin | None
    corpus: list[CorpusItem] = Field(min_length=1)
    results: list[StoredPolicy] = Field(min_length=1)

    @model_validator(mode="after")
    def corpus_matches_results(self) -> RunManifest:
        corpus_ids = [item.policy_id for item in self.corpus]
        result_ids = [item.policy_id for item in self.results]
        if corpus_ids != result_ids:
            raise ValueError("corpus and results must list the same policy ids in order")
        for item in self.results:
            if item.result.policy_id != item.policy_id:
                raise ValueError("stored result policy id does not match the corpus")
            if item.result.method != self.settings.method:
                raise ValueError("stored result method does not match the run")
        return self


def record_run(
    policies: list[PolicyDocument],
    config: AppConfig,
    method: str = DEFAULT_METHOD,
    *,
    client: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
    cache: ResultCache | None = None,
    feedback: bool = True,
    logger: logging.Logger | None = None,
    directory: str | Path | None = None,
    run_id: str | None = None,
) -> RunManifest:
    """Score the policies and write one manifest. A failed score writes nothing."""
    chosen_id = run_id if run_id is not None else str(uuid.uuid4())
    _require_run_id(chosen_id)
    if not policies:
        raise ManifestError("Run has no policies.")
    stored: list[StoredPolicy] = []
    for document in policies:
        started = time.perf_counter()
        result = run_policy(
            document.text,
            config,
            method,
            policy_id=document.policy_id,
            client=client,
            embeddings=embeddings,
            cache=cache,
            feedback=feedback,
            logger=logger,
        )
        stored.append(
            StoredPolicy(
                policy_id=document.policy_id,
                runtime_seconds=time.perf_counter() - started,
                cost_usd=_cost(result),
                result=result,
            )
        )
    manifest = RunManifest(
        run_id=chosen_id,
        settings=_settings(config, stored[0].result.method),
        prompts=_prompts(Path(config.paths.prompts_dir)),
        libraries=library_versions(),
        python_version=_python_version(),
        calibration=_calibration(Path(config.paths.calibration_dir)),
        corpus=[_corpus_item(item) for item in policies],
        results=stored,
    )
    path = _write(manifest, _directory(config, directory))
    if logger is not None:
        log_event(
            logger,
            "run_recorded",
            run_id=manifest.run_id,
            method=manifest.settings.method,
            policy_count=len(manifest.results),
            path=str(path),
        )
    return manifest


def load_run(path: str | Path) -> RunManifest:
    """Load a manifest. The recorded results do not require a model call."""
    file = Path(path)
    try:
        loaded = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Run manifest is not JSON: {file}.") from exc
    if not isinstance(loaded, dict):
        raise ManifestError(f"Run manifest must be an object: {file}.")
    try:
        return RunManifest.model_validate(loaded)
    except ValidationError as exc:
        raise ManifestError(f"Run manifest does not match the schema: {file}.") from exc


def reproduce(manifest: RunManifest) -> list[PolicyRun]:
    """Return the recorded results. This does not call a model or read the corpus."""
    return [item.result for item in manifest.results]


def run_inputs(manifest: RunManifest) -> list[PolicyDocument]:
    """Return the policy texts stored in the manifest."""
    return [PolicyDocument(policy_id=item.policy_id, text=item.text) for item in manifest.corpus]


def library_versions() -> list[LibraryVersion]:
    """Record installed libraries. Optional embedding libraries are included when present."""
    found = [_library(name) for name in _REQUIRED_LIBRARIES]
    for name in _OPTIONAL_LIBRARIES:
        try:
            found.append(LibraryVersion(name=name, version=version(name)))
        except PackageNotFoundError:
            continue
    return found


def _library(name: str) -> LibraryVersion:
    try:
        return LibraryVersion(name=name, version=version(name))
    except PackageNotFoundError as exc:
        raise ManifestError(f"Required library is not installed: {name}.") from exc


def _settings(config: AppConfig, method: Method) -> RunSettings:
    return RunSettings(
        seed=config.seed,
        truncation=config.truncation,
        method=method,
        provider=config.model.provider,
        model_id=config.model.id,
        model_name=config.model.name,
        gold_id=config.model.gold_id,
        max_output_tokens=config.model.max_output_tokens,
        embedding_model_name=config.embedding.model_name,
        embedding_revision=config.embedding.revision,
        passage_words=config.segmentation.passage_words,
        overlap_words=config.segmentation.overlap_words,
        retrieval_similarity=config.thresholds.retrieval_similarity,
        corpus_location=config.corpus_location,
        weight_mode=config.weights.mode,
        weight_value=config.weights.value,
        yes_score=config.scoring.yes_score,
        somewhat_score=config.scoring.somewhat_score,
        no_score=config.scoring.no_score,
        high_band_min=config.scoring.high_band_min,
        mid_band_min=config.scoring.mid_band_min,
    )


def _prompts(directory: Path) -> list[PromptPin]:
    registry = PromptRegistry(directory)
    pins: list[PromptPin] = []
    for folder in sorted(path for path in directory.iterdir() if path.is_dir()):
        record = registry.load(folder.name)
        pins.append(
            PromptPin(
                prompt_id=record.prompt_id,
                version=record.version,
                sha256=_sha256(record.text),
                text=record.text,
            )
        )
    if not pins:
        raise ManifestError(f"No prompts were found in {directory}.")
    return pins


def _calibration(directory: Path) -> CalibrationPin | None:
    if not calibration_versions(directory):
        return None
    record = load_calibration(directory)
    return CalibrationPin(
        version=record.version,
        threshold=record.threshold,
        label_source=record.label_source,
    )


def _corpus_item(document: PolicyDocument) -> CorpusItem:
    return CorpusItem(
        policy_id=document.policy_id,
        sha256=_sha256(document.text),
        char_count=len(document.text),
        text=document.text,
    )


def _cost(result: PolicyRun) -> float | None:
    if result.review is not None:
        usages = [item.usage for item in result.review.judgement.verdicts]
    elif result.hybrid is not None:
        usages = [item.usage for item in result.hybrid.judgement.verdicts]
    else:
        return None
    return _sum_costs(usages)


def _sum_costs(usages: list[TokenUse]) -> float | None:
    costs = [item.cost_usd for item in usages]
    if any(item is None for item in costs):
        return None
    return float(sum(item for item in costs if item is not None))


def _directory(config: AppConfig, directory: str | Path | None) -> Path:
    return Path(config.paths.runs_dir) if directory is None else Path(directory)


def _write(manifest: RunManifest, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{manifest.run_id}.json"
    if path.exists():
        raise ManifestError(f"Refusing to overwrite run manifest: {path}.")
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _require_run_id(value: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ManifestError(f"Run id must be a UUID: {value!r}.") from exc
    if str(parsed) != value:
        raise ManifestError(f"Run id must be a lowercase UUID: {value!r}.")


def _python_version() -> str:
    info = sys.version_info
    return f"{info.major}.{info.minor}.{info.micro}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

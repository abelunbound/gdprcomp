"""Repeat one method and record how often the repeats agree.

Agreement is pairwise: each pair of repeats for the same policy and criterion
counts once. Per-criterion and overall figures pool those pairs. A retrieval
repeat is labelled by its top passage, not by a Yes, Somewhat, or No verdict.
"""

from __future__ import annotations

import json
import logging
import uuid
from itertools import combinations
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gdpr_gap.config import AppConfig
from gdpr_gap.embeddings import EmbeddingClient
from gdpr_gap.legal_bert import LegalBertCriterion
from gdpr_gap.llm import LLMClient
from gdpr_gap.logging import log_event
from gdpr_gap.orchestrator import DEFAULT_METHOD, Method, PolicyDocument, PolicyRun, run_policy
from gdpr_gap.result_cache import ResultCache

MAX_REPEATS = 20
_RUN_ID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


class RepeatError(Exception):
    """Raised when repeats cannot be scored or stored."""


class PolicyLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    labels: list[str] = Field(min_length=2, max_length=MAX_REPEATS)


class CriterionAgreement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    by_policy: list[PolicyLabels] = Field(min_length=1)
    pairs: int = Field(ge=1)
    agreeing_pairs: int = Field(ge=0)
    agreement: float = Field(ge=0, le=1)
    unanimous: bool

    @model_validator(mode="after")
    def counts_match_labels(self) -> CriterionAgreement:
        pairs, agreeing = _pooled(self.by_policy)
        if pairs != self.pairs or agreeing != self.agreeing_pairs:
            raise ValueError("pair counts do not match the stored labels")
        if abs(self.agreement - (agreeing / pairs)) > 1e-9:
            raise ValueError("agreement does not match the pair counts")
        if self.unanimous != all(len(set(item.labels)) == 1 for item in self.by_policy):
            raise ValueError("unanimous does not match the stored labels")
        return self


class PolicyRepeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    runs: list[PolicyRun] = Field(min_length=2, max_length=MAX_REPEATS)

    @model_validator(mode="after")
    def runs_belong_to_this_policy(self) -> PolicyRepeat:
        if any(item.policy_id != self.policy_id for item in self.runs):
            raise ValueError("every repeat must use the policy id")
        methods = {item.method for item in self.runs}
        if len(methods) != 1:
            raise ValueError("every repeat must use the same method")
        return self


class RepeatReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repeat_id: str = Field(pattern=_RUN_ID)
    method: Method
    repeats: int = Field(ge=2, le=MAX_REPEATS)
    policies: list[PolicyRepeat] = Field(min_length=1)
    criteria: list[CriterionAgreement] = Field(min_length=22, max_length=22)
    overall_pairs: int = Field(ge=1)
    overall_agreeing_pairs: int = Field(ge=0)
    overall_agreement: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def overall_matches_criteria(self) -> RepeatReport:
        pairs = sum(item.pairs for item in self.criteria)
        agreeing = sum(item.agreeing_pairs for item in self.criteria)
        if pairs != self.overall_pairs or agreeing != self.overall_agreeing_pairs:
            raise ValueError("overall pair counts do not match the criteria")
        if abs(self.overall_agreement - (agreeing / pairs)) > 1e-9:
            raise ValueError("overall agreement does not match the pair counts")
        ids = [item.policy_id for item in self.policies]
        if len(set(ids)) != len(ids):
            raise ValueError("each policy id must be unique")
        for policy in self.policies:
            if len(policy.runs) != self.repeats or policy.runs[0].method != self.method:
                raise ValueError("stored repeats do not match the report")
        return self


def record_repeats(
    policies: list[PolicyDocument],
    config: AppConfig,
    method: str = DEFAULT_METHOD,
    *,
    repeats: int,
    client: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
    cache: ResultCache | None = None,
    feedback: bool = True,
    logger: logging.Logger | None = None,
    directory: str | Path | None = None,
    repeat_id: str | None = None,
) -> RepeatReport:
    """Score each policy N times and write the agreement report."""
    chosen_id = repeat_id if repeat_id is not None else str(uuid.uuid4())
    _require_id(chosen_id)
    _require_repeats(repeats)
    if not policies:
        raise RepeatError("Repeat run has no policies.")
    recorded = [
        PolicyRepeat(
            policy_id=document.policy_id,
            runs=[
                run_policy(
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
                for _repeat in range(repeats)
            ],
        )
        for document in policies
    ]
    report = _report(chosen_id, recorded)
    path = _write(report, _directory(config, directory))
    if logger is not None:
        log_event(
            logger,
            "repeats_recorded",
            repeat_id=report.repeat_id,
            method=report.method,
            repeats=report.repeats,
            overall_agreement=report.overall_agreement,
            path=str(path),
        )
    return report


def load_repeats(path: str | Path) -> RepeatReport:
    """Load a stored repeat report. Agreement is already in the file."""
    file = Path(path)
    try:
        loaded = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RepeatError(f"Repeat report is not JSON: {file}.") from exc
    if not isinstance(loaded, dict):
        raise RepeatError(f"Repeat report must be an object: {file}.")
    try:
        return RepeatReport.model_validate(loaded)
    except ValidationError as exc:
        raise RepeatError(f"Repeat report does not match the schema: {file}.") from exc


def agreement_report(report: RepeatReport) -> str:
    """Render per-criterion and overall agreement as text."""
    lines = [
        f"Method: {report.method}",
        f"Repeats: {report.repeats}",
        (
            f"Overall agreement: {report.overall_agreeing_pairs}/{report.overall_pairs}"
            f" ({report.overall_agreement:.3f})"
        ),
    ]
    for row in report.criteria:
        lines.append(f"{row.criterion_id}: {row.agreeing_pairs}/{row.pairs} ({row.agreement:.3f})")
    return "\n".join(lines) + "\n"


def _report(repeat_id: str, policies: list[PolicyRepeat]) -> RepeatReport:
    method = policies[0].runs[0].method
    repeats = len(policies[0].runs)
    order = [criterion_id for criterion_id, _label in _labels(policies[0].runs[0])]
    criteria = [_criterion(criterion_id, policies) for criterion_id in order]
    overall_pairs = sum(item.pairs for item in criteria)
    overall_agreeing = sum(item.agreeing_pairs for item in criteria)
    return RepeatReport(
        repeat_id=repeat_id,
        method=method,
        repeats=repeats,
        policies=policies,
        criteria=criteria,
        overall_pairs=overall_pairs,
        overall_agreeing_pairs=overall_agreeing,
        overall_agreement=overall_agreeing / overall_pairs,
    )


def _criterion(criterion_id: str, policies: list[PolicyRepeat]) -> CriterionAgreement:
    by_policy = [
        PolicyLabels(policy_id=policy.policy_id, labels=_labels_for(policy.runs, criterion_id))
        for policy in policies
    ]
    pairs, agreeing = _pooled(by_policy)
    return CriterionAgreement(
        criterion_id=criterion_id,
        by_policy=by_policy,
        pairs=pairs,
        agreeing_pairs=agreeing,
        agreement=agreeing / pairs,
        unanimous=all(len(set(item.labels)) == 1 for item in by_policy),
    )


def _labels_for(runs: list[PolicyRun], criterion_id: str) -> list[str]:
    found: list[str] = []
    for run in runs:
        matches = [label for item_id, label in _labels(run) if item_id == criterion_id]
        if len(matches) != 1:
            raise RepeatError(f"Criterion {criterion_id} was not scored once in a repeat.")
        found.append(matches[0])
    return found


def _labels(run: PolicyRun) -> list[tuple[str, str]]:
    if run.review is not None:
        verdicts = run.review.judgement.verdicts
        return [(item.judgement.criterion_id, item.judgement.verdict) for item in verdicts]
    if run.hybrid is not None:
        verdicts = run.hybrid.judgement.verdicts
        return [(item.judgement.criterion_id, item.judgement.verdict) for item in verdicts]
    if run.retrieval is None:
        raise RepeatError("A repeat result has no verdicts and no retrieval signal.")
    return [(item.criterion_id, _retrieval_label(item)) for item in run.retrieval.criteria]


def _retrieval_label(row: LegalBertCriterion) -> str:
    top = row.passages[0]
    return f"{top.index}:{top.similarity:.10f}"


def _pooled(rows: list[PolicyLabels]) -> tuple[int, int]:
    pairs = 0
    agreeing = 0
    for row in rows:
        for left, right in combinations(row.labels, 2):
            pairs += 1
            if left == right:
                agreeing += 1
    return pairs, agreeing


def _directory(config: AppConfig, directory: str | Path | None) -> Path:
    if directory is not None:
        return Path(directory)
    return Path(config.paths.runs_dir) / "repeats"


def _write(report: RepeatReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.repeat_id}.json"
    if path.exists():
        raise RepeatError(f"Refusing to overwrite repeat report: {path}.")
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _require_repeats(repeats: int) -> None:
    if repeats < 2 or repeats > MAX_REPEATS:
        raise RepeatError(f"Repeats must be from 2 to {MAX_REPEATS}.")


def _require_id(value: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise RepeatError(f"Repeat id must be a UUID: {value!r}.") from exc
    if str(parsed) != value:
        raise RepeatError(f"Repeat id must be a lowercase UUID: {value!r}.")

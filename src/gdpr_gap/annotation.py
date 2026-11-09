"""Blind double rating and adjudication.

Each annotator works in a separate set and cannot read the other annotator's
verdict. Merging the two sets is what the adjudicator sees. The adjudicated
verdict is the final label. Every record is model-produced.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Verdict = Literal["Yes", "Somewhat", "No"]
LABEL_SOURCE: Literal["model-produced"] = "model-produced"


class AnnotationError(Exception):
    """Raised when a label or adjudication cannot be recorded."""


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    annotator_id: str = Field(min_length=1)
    verdict: Verdict
    notes: str = ""
    timestamp: datetime
    label_source: Literal["model-produced"] = LABEL_SOURCE

    @field_validator("timestamp")
    @classmethod
    def timestamp_has_a_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value


class Adjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    adjudicator_id: str = Field(min_length=1)
    adjudicated_verdict: Verdict
    notes: str = ""
    timestamp: datetime
    annotator_ids: list[str] = Field(min_length=2, max_length=2)
    label_source: Literal["model-produced"] = LABEL_SOURCE

    @field_validator("timestamp")
    @classmethod
    def timestamp_has_a_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value

    @field_validator("annotator_ids")
    @classmethod
    def two_distinct_annotators(cls, value: list[str]) -> list[str]:
        if len(set(value)) != 2:
            raise ValueError("adjudication must cite two distinct annotators")
        return sorted(value)


class AnnotationSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_source: Literal["model-produced"] = LABEL_SOURCE
    judge_model_id: str | None = None
    annotations: list[Annotation] = Field(default_factory=list)
    adjudications: list[Adjudication] = Field(default_factory=list)

    @field_validator("judge_model_id")
    @classmethod
    def judge_id_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("judge_model_id must not be blank")
        return value


def empty_set(judge_model_id: str | None = None) -> AnnotationSet:
    """Start a set whose labels cannot use the judge model id."""
    return AnnotationSet(judge_model_id=judge_model_id)


def add_annotation(collection: AnnotationSet, item: Annotation) -> AnnotationSet:
    """Record one annotator's label. The other annotator's rows stay unchanged."""
    _reject_judge(collection.judge_model_id, item.annotator_id, "Annotator")
    key = (item.policy_id, item.criterion_id)
    existing = _pair(collection, key)
    if any(row.annotator_id == item.annotator_id for row in existing):
        raise AnnotationError("This annotator already labelled this policy and criterion.")
    if len(existing) >= 2:
        raise AnnotationError("Two annotators have already labelled this policy and criterion.")
    return _rebuild(collection, [*collection.annotations, item], list(collection.adjudications))


def blind_view(collection: AnnotationSet, annotator_id: str) -> AnnotationSet:
    """Return only this annotator's labels, with no adjudication and no other rater."""
    own = [item for item in collection.annotations if item.annotator_id == annotator_id]
    if not own:
        raise AnnotationError(f"No labels are stored for annotator {annotator_id}.")
    return AnnotationSet(judge_model_id=collection.judge_model_id, annotations=own)


def merge_annotations(parts: list[AnnotationSet]) -> AnnotationSet:
    """Combine independent annotator sets for adjudication."""
    if len(parts) < 2:
        raise AnnotationError("Merging needs at least two annotator sets.")
    judge_ids = {part.judge_model_id for part in parts}
    if len(judge_ids) != 1:
        raise AnnotationError("Annotator sets name different judge models.")
    combined = empty_set(parts[0].judge_model_id)
    for part in parts:
        if part.adjudications:
            raise AnnotationError("A blind annotator set cannot contain adjudications.")
        for item in part.annotations:
            combined = add_annotation(combined, item)
    return combined


def adjudicate(
    collection: AnnotationSet,
    *,
    adjudicator_id: str,
    policy_id: str,
    criterion_id: str,
    verdict: Verdict,
    notes: str,
    timestamp: datetime,
) -> AnnotationSet:
    """Record the final verdict for one double-rated policy and criterion."""
    key = (policy_id, criterion_id)
    pair = _pair(collection, key)
    if len(pair) != 2:
        raise AnnotationError("Adjudication needs two independent labels.")
    if _already_final(collection, key):
        raise AnnotationError("This policy and criterion already has a final label.")
    annotator_ids = sorted({item.annotator_id for item in pair})
    _reject_judge(collection.judge_model_id, adjudicator_id, "Adjudicator")
    if adjudicator_id in annotator_ids:
        raise AnnotationError("The adjudicator must not be one of the two annotators.")
    record = Adjudication(
        policy_id=policy_id,
        criterion_id=criterion_id,
        adjudicator_id=adjudicator_id,
        adjudicated_verdict=verdict,
        notes=notes,
        timestamp=timestamp,
        annotator_ids=annotator_ids,
    )
    return _rebuild(collection, list(collection.annotations), [*collection.adjudications, record])


def adopt_agreement(
    collection: AnnotationSet,
    *,
    adjudicator_id: str,
    timestamp: datetime,
    notes: str = "Raters agreed.",
) -> AnnotationSet:
    """Adjudicate every pair whose two verdicts match. Disagreements stay open."""
    updated = collection
    adopted = 0
    seen: set[tuple[str, str]] = set()
    for item in collection.annotations:
        key = (item.policy_id, item.criterion_id)
        if key in seen:
            continue
        seen.add(key)
        pair = _pair(collection, key)
        if len(pair) != 2 or pair[0].verdict != pair[1].verdict:
            continue
        if _already_final(updated, key):
            continue
        updated = adjudicate(
            updated,
            adjudicator_id=adjudicator_id,
            policy_id=key[0],
            criterion_id=key[1],
            verdict=pair[0].verdict,
            notes=notes,
            timestamp=timestamp,
        )
        adopted += 1
    if adopted == 0:
        raise AnnotationError("No agreed pair is waiting for adjudication.")
    return updated


def final_labels(collection: AnnotationSet) -> list[Adjudication]:
    """Return the adjudicated verdicts. Unadjudicated pairs are omitted."""
    return list(collection.adjudications)


def export_annotations(collection: AnnotationSet) -> str:
    """Export a set as JSON."""
    payload = collection.model_dump(mode="json")
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def export_blind(collection: AnnotationSet, annotator_id: str) -> str:
    """Export one annotator's labels. The other rater and the final labels are omitted."""
    return export_annotations(blind_view(collection, annotator_id))


def import_annotations(payload: str) -> AnnotationSet:
    """Load a JSON set. Expert label_source is rejected by the schema."""
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AnnotationError("Annotation file is not JSON.") from exc
    if not isinstance(loaded, dict):
        raise AnnotationError("Annotation file must be an object.")
    try:
        return AnnotationSet.model_validate(loaded)
    except ValidationError as exc:
        raise AnnotationError("Annotation file does not match the schema.") from exc


def _already_final(collection: AnnotationSet, key: tuple[str, str]) -> bool:
    return any((item.policy_id, item.criterion_id) == key for item in collection.adjudications)


def _pair(collection: AnnotationSet, key: tuple[str, str]) -> list[Annotation]:
    return [item for item in collection.annotations if (item.policy_id, item.criterion_id) == key]


def _rebuild(
    collection: AnnotationSet,
    annotations: list[Annotation],
    adjudications: list[Adjudication],
) -> AnnotationSet:
    return AnnotationSet(
        judge_model_id=collection.judge_model_id,
        annotations=annotations,
        adjudications=adjudications,
    )


def _reject_judge(judge_model_id: str | None, actor_id: str, role: str) -> None:
    if judge_model_id is not None and actor_id == judge_model_id:
        raise AnnotationError(f"{role} must differ from the judge model.")

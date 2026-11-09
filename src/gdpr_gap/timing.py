"""Labelling time and the manual-review baseline.

Each entry is one policy and criterion. The entries roll up to a baseline,
and that baseline is what “time saved” subtracts an automated run from.
The times are model-produced labelling times.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

LABEL_SOURCE: Literal["model-produced"] = "model-produced"


class TimingError(Exception):
    """Raised when labelling time cannot be recorded or rolled up."""


class TimeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    annotator_id: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    label_source: Literal["model-produced"] = LABEL_SOURCE

    @field_validator("started_at", "finished_at")
    @classmethod
    def timestamp_has_a_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def finished_after_started(self) -> TimeEntry:
        if self.finished_at <= self.started_at:
            raise ValueError("finished_at must be after started_at")
        return self

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


class TimeLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_source: Literal["model-produced"] = LABEL_SOURCE
    entries: list[TimeEntry] = Field(default_factory=list)


class ManualReviewBaseline(BaseModel):
    """Rolled-up labelling time. This is the figure used in time saved."""

    model_config = ConfigDict(extra="forbid")

    label_source: Literal["model-produced"] = LABEL_SOURCE
    entry_count: int = Field(ge=1)
    total_seconds: float = Field(gt=0)
    seconds_per_label: float = Field(gt=0)

    @model_validator(mode="after")
    def rate_matches_total(self) -> ManualReviewBaseline:
        expected = self.total_seconds / self.entry_count
        if abs(self.seconds_per_label - expected) > 1e-6:
            raise ValueError("seconds_per_label must be total_seconds divided by entry_count")
        return self


class TimeSaved(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_source: Literal["model-produced"] = LABEL_SOURCE
    baseline_seconds: float = Field(gt=0)
    automated_seconds: float = Field(ge=0)
    saved_seconds: float

    @model_validator(mode="after")
    def saved_is_baseline_minus_automated(self) -> TimeSaved:
        expected = self.baseline_seconds - self.automated_seconds
        if abs(self.saved_seconds - expected) > 1e-6:
            raise ValueError("saved_seconds must be the baseline minus the automated run")
        return self


def record_time(log: TimeLog, entry: TimeEntry) -> TimeLog:
    """Append one labelling interval. The same policy and criterion cannot be timed twice."""
    key = (entry.policy_id, entry.criterion_id, entry.annotator_id)
    for existing in log.entries:
        if (existing.policy_id, existing.criterion_id, existing.annotator_id) == key:
            raise TimingError(
                "Labelling time for this policy, criterion, and annotator is already recorded."
            )
    return TimeLog(entries=[*log.entries, entry])


def manual_review_baseline(log: TimeLog) -> ManualReviewBaseline:
    """Roll every entry into the manual-review baseline."""
    if not log.entries:
        raise TimingError("Cannot build a manual-review baseline with no time entries.")
    total = sum(item.duration_seconds for item in log.entries)
    count = len(log.entries)
    return ManualReviewBaseline(
        entry_count=count,
        total_seconds=total,
        seconds_per_label=total / count,
    )


def time_saved(baseline: ManualReviewBaseline, automated_seconds: float) -> TimeSaved:
    """Subtract an automated run from the manual-review baseline."""
    if automated_seconds < 0:
        raise TimingError("Automated time cannot be negative.")
    return TimeSaved(
        baseline_seconds=baseline.total_seconds,
        automated_seconds=automated_seconds,
        saved_seconds=baseline.total_seconds - automated_seconds,
    )


def time_saved_report(baseline: ManualReviewBaseline, saved: TimeSaved) -> str:
    """Report the baseline and the time saved against an automated run."""
    return (
        "Manual-review baseline: "
        f"{baseline.total_seconds:.1f} seconds across {baseline.entry_count} labels "
        f"({baseline.seconds_per_label:.1f} seconds per label). "
        "This baseline is model-produced labelling time.\n"
        f"Automated run: {saved.automated_seconds:.1f} seconds.\n"
        f"Time saved: {saved.saved_seconds:.1f} seconds.\n"
    )


def export_times(log: TimeLog) -> str:
    """Export the time log as JSON."""
    return json.dumps(log.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def import_times(payload: str) -> TimeLog:
    """Load a time log. Expert label_source is rejected by the schema."""
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise TimingError("Time log is not JSON.") from exc
    if not isinstance(loaded, dict):
        raise TimingError("Time log must be an object.")
    try:
        return TimeLog.model_validate(loaded)
    except ValidationError as exc:
        raise TimingError("Time log does not match the schema.") from exc

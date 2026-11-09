import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from gdpr_gap.timing import (
    TimeEntry,
    TimeLog,
    TimingError,
    export_times,
    import_times,
    manual_review_baseline,
    record_time,
    time_saved,
    time_saved_report,
)

START = datetime(2026, 11, 9, 21, 0, tzinfo=UTC)
SONNET = "claude-sonnet-5-5"


def _entry(seconds: float, criterion_id: str) -> TimeEntry:
    return TimeEntry(
        policy_id="745",
        criterion_id=criterion_id,
        annotator_id=SONNET,
        started_at=START,
        finished_at=START + timedelta(seconds=seconds),
    )


def test_time_entries_roll_up_into_the_baseline_used_for_time_saved() -> None:
    log = record_time(TimeLog(), _entry(10, "data_processing_records"))
    log = record_time(log, _entry(20, "purpose_limitation"))
    restored = import_times(export_times(log))
    baseline = manual_review_baseline(restored)
    saved = time_saved(baseline, 12)
    report = time_saved_report(baseline, saved)

    assert baseline.entry_count == 2
    assert baseline.total_seconds == 30
    assert baseline.seconds_per_label == 15
    assert baseline.label_source == "model-produced"
    assert saved.baseline_seconds == 30
    assert saved.automated_seconds == 12
    assert saved.saved_seconds == 18
    assert "Manual-review baseline: 30.0 seconds across 2 labels" in report
    assert "Time saved: 18.0 seconds." in report
    assert "model-produced" in report


def test_invalid_times_are_rejected() -> None:
    logged = record_time(TimeLog(), _entry(10, "data_processing_records"))
    with pytest.raises(TimingError, match="already recorded"):
        record_time(logged, _entry(5, "data_processing_records"))
    with pytest.raises(ValidationError, match="finished_at must be after started_at"):
        _entry(0, "purpose_limitation")
    with pytest.raises(TimingError, match="no time entries"):
        manual_review_baseline(TimeLog())
    with pytest.raises(TimingError, match="negative"):
        time_saved(manual_review_baseline(logged), -1)
    with pytest.raises(TimingError, match="schema"):
        payload = json.loads(export_times(logged))
        payload["label_source"] = "expert"
        import_times(json.dumps(payload))

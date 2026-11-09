"""The single definition of a compliance gap.

A gap is any verdict other than an exact Yes. Somewhat and No are both gaps.
Metrics, error examples, and the Dash views must call is_gap rather than
comparing verdict strings themselves.
"""

from __future__ import annotations

from collections.abc import Mapping


def is_gap(verdict: str) -> bool:
    """Return whether this verdict is a gap."""
    return verdict != "Yes"


def gap_ids(verdicts: Mapping[str, str]) -> list[str]:
    """Return criterion ids whose verdicts are gaps, in mapping order."""
    return [criterion_id for criterion_id, verdict in verdicts.items() if is_gap(verdict)]

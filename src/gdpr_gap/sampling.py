"""Draw the gold-set sample from a cached corpus.

The pre window keeps the latest snapshot per site. The post window keeps the
earliest. Sites that have both are drawn first. The 2018 buffer is never
sampled. The same seed returns the same snapshot ids.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gdpr_gap.config import SamplingConfig
from gdpr_gap.corpus import PolicyCorpus, PolicyRecord
from gdpr_gap.logging import log_event

GOLD_PER_WINDOW = 10
DESCRIPTIVE_PER_WINDOW = 40
NOT_VALIDATION: Literal["not validation unless labelled"] = "not validation unless labelled"
WindowName = Literal["pre", "post", "buffer", "outside"]
_LATER_PRE = "excluded: a later pre-window snapshot was kept for this site"
_EARLIER_POST = "excluded: an earlier post-window snapshot was kept for this site"
_BUFFER = "excluded: snapshot date is inside the buffer window"
_OUTSIDE = "excluded: snapshot date is outside the pre, buffer, and post windows"
_NOT_DRAWN = "excluded: site was not drawn for this seed"
_RESERVED = "excluded: site is reserved for the gold set"
_PAIRED_PRE = "included: latest pre-window snapshot for a site drawn in both windows"
_PAIRED_POST = "included: earliest post-window snapshot for a site drawn in both windows"
_FILL_PRE = "included: latest pre-window snapshot drawn to fill the pre sample"
_FILL_POST = "included: earliest post-window snapshot drawn to fill the post sample"


class SamplingError(Exception):
    """Raised when a gold-set sample cannot be drawn."""


class SelectedSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    domain: str
    url: str
    snapshot_date: date
    window: Literal["pre", "post"]
    paired: bool
    reason: str


class SampleDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    domain: str
    snapshot_date: date
    action: Literal["included", "excluded"]
    window: WindowName
    reason: str


class GoldSample(BaseModel):
    """A gold-set sample frame. Labels are not included and are not ground truth."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["gold"] = "gold"
    seed: int = Field(ge=0)
    per_window: int = Field(ge=1)
    domain_suffix: str
    pre_start: date
    pre_end: date
    buffer_start: date
    buffer_end: date
    post_start: date
    post_end: date
    paired_site_count: int = Field(ge=0)
    policy_ids: list[str]
    selected: list[SelectedSnapshot]
    decisions: list[SampleDecision]

    def write_manifest(self, path: str | Path) -> Path:
        """Write the sample and every inclusion or exclusion reason."""
        return _write_manifest(path, self.model_dump(mode="json"))


class DescriptiveSample(BaseModel):
    """A descriptive sample. It is not a validation set unless it is later labelled."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["descriptive"] = "descriptive"
    notice: Literal["not validation unless labelled"] = NOT_VALIDATION
    seed: int = Field(ge=0)
    per_window: int = Field(ge=1)
    domain_suffix: str
    pre_start: date
    pre_end: date
    buffer_start: date
    buffer_end: date
    post_start: date
    post_end: date
    paired_site_count: int = Field(ge=0)
    reserved_domains: list[str]
    policy_ids: list[str]
    selected: list[SelectedSnapshot]
    decisions: list[SampleDecision]

    def write_manifest(self, path: str | Path) -> Path:
        """Write the descriptive sample. The notice forbids treating it as validation."""
        return _write_manifest(path, self.model_dump(mode="json"))


def sample_gold_set(
    corpus: PolicyCorpus,
    sampling: SamplingConfig,
    *,
    seed: int,
    per_window: int = GOLD_PER_WINDOW,
    manifest_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> GoldSample:
    """Sample paired sites first, then fill either window if fewer than `per_window` pairs exist."""
    _check_request(corpus, sampling, seed, per_window)
    selected, decisions, paired_count = _assemble(corpus, sampling, seed, per_window, set())
    sample = _gold_sample(sampling, seed, per_window, selected, decisions, paired_count)
    if manifest_path is not None:
        sample.write_manifest(manifest_path)
    if logger is not None:
        _log_sample(logger, sample, "gold_sample")
    return sample


def sample_descriptive_set(
    corpus: PolicyCorpus,
    sampling: SamplingConfig,
    *,
    seed: int,
    per_window: int = DESCRIPTIVE_PER_WINDOW,
    reserved_domains: set[str] | None = None,
    manifest_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> DescriptiveSample:
    """Draw the descriptive corpus. Gold-set sites passed in `reserved_domains` are left out."""
    _check_request(corpus, sampling, seed, per_window)
    reserved = set(reserved_domains or ())
    selected, decisions, paired_count = _assemble(corpus, sampling, seed, per_window, reserved)
    sample = DescriptiveSample(
        seed=seed,
        per_window=per_window,
        domain_suffix=sampling.domain_suffix,
        pre_start=sampling.pre_start,
        pre_end=sampling.pre_end,
        buffer_start=sampling.buffer_start,
        buffer_end=sampling.buffer_end,
        post_start=sampling.post_start,
        post_end=sampling.post_end,
        paired_site_count=paired_count,
        reserved_domains=sorted(reserved),
        policy_ids=[item.snapshot_id for item in selected],
        selected=selected,
        decisions=decisions,
    )
    if manifest_path is not None:
        sample.write_manifest(manifest_path)
    if logger is not None:
        _log_sample(logger, sample, "descriptive_sample")
    return sample


def load_gold_manifest(path: str | Path) -> GoldSample:
    """Load a gold manifest. A descriptive manifest is rejected."""
    payload = _read_manifest(path)
    notice = str(payload.get("notice", ""))
    if payload.get("role") != "gold" or NOT_VALIDATION in notice:
        raise SamplingError(
            f"{path} is not a gold-set manifest. "
            "A descriptive sample is not validation unless labelled."
        )
    return GoldSample.model_validate(payload)


def _check_request(
    corpus: PolicyCorpus, sampling: SamplingConfig, seed: int, per_window: int
) -> None:
    if per_window < 1:
        raise SamplingError("per_window must be at least 1")
    if seed < 0:
        raise SamplingError("seed must be zero or positive")
    if corpus.domain_suffix != sampling.domain_suffix:
        raise SamplingError(
            f"Corpus suffix {corpus.domain_suffix!r} does not match {sampling.domain_suffix!r}."
        )


def _assemble(
    corpus: PolicyCorpus,
    sampling: SamplingConfig,
    seed: int,
    per_window: int,
    reserved: set[str],
) -> tuple[list[SelectedSnapshot], list[SampleDecision], int]:
    grouped = _group_by_window(corpus, sampling)
    pre_choice, pre_rejected = _choose_site_snapshots(grouped["pre"], latest=True)
    post_choice, post_rejected = _choose_site_snapshots(grouped["post"], latest=False)
    available_pre = {
        domain: record for domain, record in pre_choice.items() if domain not in reserved
    }
    available_post = {
        domain: record for domain, record in post_choice.items() if domain not in reserved
    }
    paired_domains = sorted(set(available_pre) & set(available_post))
    drawn_pre, drawn_post = _draw_sites(
        paired_domains, available_pre, available_post, seed, per_window
    )
    decisions = _decisions(
        corpus,
        sampling,
        pre_choice,
        post_choice,
        pre_rejected,
        post_rejected,
        drawn_pre,
        drawn_post,
        paired_domains,
        reserved,
    )
    selected = _selected(pre_choice, post_choice, drawn_pre, drawn_post, paired_domains)
    return selected, decisions, len(paired_domains)


def _gold_sample(
    sampling: SamplingConfig,
    seed: int,
    per_window: int,
    selected: list[SelectedSnapshot],
    decisions: list[SampleDecision],
    paired_count: int,
) -> GoldSample:
    return GoldSample(
        seed=seed,
        per_window=per_window,
        domain_suffix=sampling.domain_suffix,
        pre_start=sampling.pre_start,
        pre_end=sampling.pre_end,
        buffer_start=sampling.buffer_start,
        buffer_end=sampling.buffer_end,
        post_start=sampling.post_start,
        post_end=sampling.post_end,
        paired_site_count=paired_count,
        policy_ids=[item.snapshot_id for item in selected],
        selected=selected,
        decisions=decisions,
    )


def _write_manifest(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _read_manifest(path: str | Path) -> dict[str, object]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SamplingError(f"{path} is not a sampling manifest.")
    return loaded


def _group_by_window(
    corpus: PolicyCorpus, sampling: SamplingConfig
) -> dict[WindowName, list[PolicyRecord]]:
    grouped: dict[WindowName, list[PolicyRecord]] = {
        "pre": [],
        "post": [],
        "buffer": [],
        "outside": [],
    }
    for record in corpus.records:
        if not record.domain.endswith(sampling.domain_suffix):
            raise SamplingError(
                f"Snapshot {record.snapshot_id} domain {record.domain!r} does not end with "
                f"{sampling.domain_suffix!r}."
            )
        grouped[_window_for(record.snapshot_date, sampling)].append(record)
    return grouped


def _window_for(snapshot_date: date, sampling: SamplingConfig) -> WindowName:
    if sampling.pre_start <= snapshot_date <= sampling.pre_end:
        return "pre"
    if sampling.buffer_start <= snapshot_date <= sampling.buffer_end:
        return "buffer"
    if sampling.post_start <= snapshot_date <= sampling.post_end:
        return "post"
    return "outside"


def _choose_site_snapshots(
    records: list[PolicyRecord], *, latest: bool
) -> tuple[dict[str, PolicyRecord], list[PolicyRecord]]:
    by_domain: dict[str, list[PolicyRecord]] = defaultdict(list)
    for record in records:
        by_domain[record.domain].append(record)
    chosen: dict[str, PolicyRecord] = {}
    rejected: list[PolicyRecord] = []
    for domain in sorted(by_domain):
        group = by_domain[domain]
        pick = _pick(group, latest=latest)
        chosen[domain] = pick
        rejected.extend(record for record in group if record.snapshot_id != pick.snapshot_id)
    return chosen, rejected


def _pick(records: list[PolicyRecord], *, latest: bool) -> PolicyRecord:
    if latest:
        return max(records, key=lambda record: (record.snapshot_date, int(record.snapshot_id)))
    return min(records, key=lambda record: (record.snapshot_date, int(record.snapshot_id)))


def _draw_sites(
    paired_domains: list[str],
    pre_choice: dict[str, PolicyRecord],
    post_choice: dict[str, PolicyRecord],
    seed: int,
    per_window: int,
) -> tuple[list[str], list[str]]:
    generator = random.Random(seed)
    paired = list(paired_domains)
    generator.shuffle(paired)
    drawn_pairs = paired[:per_window]
    pre_only = sorted(set(pre_choice) - set(drawn_pairs))
    post_only = sorted(set(post_choice) - set(drawn_pairs))
    generator.shuffle(pre_only)
    generator.shuffle(post_only)
    drawn_pre = drawn_pairs + pre_only[: max(per_window - len(drawn_pairs), 0)]
    drawn_post = drawn_pairs + post_only[: max(per_window - len(drawn_pairs), 0)]
    if len(drawn_pre) < per_window or len(drawn_post) < per_window:
        raise SamplingError(
            f"Need {per_window} pre and {per_window} post sites. "
            f"Found {len(pre_choice)} pre, {len(post_choice)} post, "
            f"and {len(paired_domains)} paired."
        )
    return drawn_pre[:per_window], drawn_post[:per_window]


def _decisions(
    corpus: PolicyCorpus,
    sampling: SamplingConfig,
    pre_choice: dict[str, PolicyRecord],
    post_choice: dict[str, PolicyRecord],
    pre_rejected: list[PolicyRecord],
    post_rejected: list[PolicyRecord],
    drawn_pre: list[str],
    drawn_post: list[str],
    paired_domains: list[str],
    reserved: set[str],
) -> list[SampleDecision]:
    drawn = {"pre": set(drawn_pre), "post": set(drawn_post)}
    paired = set(paired_domains)
    rejected_reason = {
        record.snapshot_id: _LATER_PRE if record in pre_rejected else _EARLIER_POST
        for record in [*pre_rejected, *post_rejected]
    }
    decisions: list[SampleDecision] = []
    for record in sorted(corpus.records, key=lambda item: int(item.snapshot_id)):
        window = _window_for(record.snapshot_date, sampling)
        action, reason = _decision_for(
            record, window, pre_choice, post_choice, drawn, paired, rejected_reason, reserved
        )
        decisions.append(
            SampleDecision(
                snapshot_id=record.snapshot_id,
                domain=record.domain,
                snapshot_date=record.snapshot_date,
                action=action,
                window=window,
                reason=reason,
            )
        )
    return decisions


def _decision_for(
    record: PolicyRecord,
    window: WindowName,
    pre_choice: dict[str, PolicyRecord],
    post_choice: dict[str, PolicyRecord],
    drawn: dict[str, set[str]],
    paired: set[str],
    rejected_reason: dict[str, str],
    reserved: set[str],
) -> tuple[Literal["included", "excluded"], str]:
    if window == "buffer":
        return "excluded", _BUFFER
    if window == "outside":
        return "excluded", _OUTSIDE
    if record.snapshot_id in rejected_reason:
        return "excluded", rejected_reason[record.snapshot_id]
    if record.domain in reserved and window in ("pre", "post"):
        return "excluded", _RESERVED
    chosen = pre_choice if window == "pre" else post_choice
    kept = chosen.get(record.domain)
    if kept is None or kept.snapshot_id != record.snapshot_id:
        fallback = rejected_reason.get(record.snapshot_id, "excluded: snapshot was not chosen")
        return "excluded", fallback
    if record.domain not in drawn[window]:
        return "excluded", _NOT_DRAWN
    if record.domain in paired:
        return "included", _PAIRED_PRE if window == "pre" else _PAIRED_POST
    return "included", _FILL_PRE if window == "pre" else _FILL_POST


def _selected(
    pre_choice: dict[str, PolicyRecord],
    post_choice: dict[str, PolicyRecord],
    drawn_pre: list[str],
    drawn_post: list[str],
    paired_domains: list[str],
) -> list[SelectedSnapshot]:
    paired = set(paired_domains)
    selected: list[SelectedSnapshot] = []
    for domain in drawn_pre:
        selected.append(_selected_row(pre_choice[domain], "pre", domain in paired))
    for domain in drawn_post:
        selected.append(_selected_row(post_choice[domain], "post", domain in paired))
    return selected


def _selected_row(
    record: PolicyRecord, window: Literal["pre", "post"], paired: bool
) -> SelectedSnapshot:
    if paired and window == "pre":
        reason = _PAIRED_PRE
    elif paired:
        reason = _PAIRED_POST
    elif window == "pre":
        reason = _FILL_PRE
    else:
        reason = _FILL_POST
    return SelectedSnapshot(
        snapshot_id=record.snapshot_id,
        domain=record.domain,
        url=record.url,
        snapshot_date=record.snapshot_date,
        window=window,
        paired=paired,
        reason=reason,
    )


def _log_sample(logger: logging.Logger, sample: GoldSample | DescriptiveSample, event: str) -> None:
    included = [item for item in sample.decisions if item.action == "included"]
    excluded = len(sample.decisions) - len(included)
    log_event(
        logger,
        event,
        seed=sample.seed,
        included=len(included),
        excluded=excluded,
        paired_sites=sample.paired_site_count,
    )
    for item in sample.selected:
        log_event(
            logger,
            f"{event}_included",
            snapshot_id=item.snapshot_id,
            domain=item.domain,
            window=item.window,
            reason=item.reason,
        )

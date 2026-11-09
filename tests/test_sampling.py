import json
import logging
from datetime import date
from pathlib import Path

import pytest

from gdpr_gap.config import SamplingConfig
from gdpr_gap.corpus import PolicyCorpus, PolicyRecord
from gdpr_gap.sampling import SamplingError, sample_gold_set


def _sampling() -> SamplingConfig:
    return SamplingConfig(
        domain_suffix=".co.uk",
        pre_start=date(2017, 1, 1),
        pre_end=date(2017, 12, 31),
        buffer_start=date(2018, 1, 1),
        buffer_end=date(2018, 12, 31),
        post_start=date(2019, 1, 1),
        post_end=date(2019, 12, 31),
    )


def _record(snapshot_id: str, domain: str, snapshot_date: date) -> PolicyRecord:
    stamp = snapshot_date.strftime("%Y%m%d")
    return PolicyRecord(
        snapshot_id=snapshot_id,
        domain=domain,
        url=f"{domain}/privacy",
        snapshot_url=f"https://web.archive.org/web/{stamp}120000id_/http://{domain}/privacy",
        snapshot_date=snapshot_date,
        text=f"Policy text {snapshot_id}.",
    )


def _corpus(records: list[PolicyRecord]) -> PolicyCorpus:
    return PolicyCorpus(domain_suffix=".co.uk", records=records)


def _paired_corpus() -> PolicyCorpus:
    records: list[PolicyRecord] = []
    next_id = 1
    for index in range(12):
        domain = f"site{index:02d}.co.uk"
        records.append(_record(str(next_id), domain, date(2017, 3, 1)))
        next_id += 1
        records.append(_record(str(next_id), domain, date(2017, 11, 1)))
        next_id += 1
        records.append(_record(str(next_id), domain, date(2018, 5, 25)))
        next_id += 1
        records.append(_record(str(next_id), domain, date(2019, 2, 1)))
        next_id += 1
        records.append(_record(str(next_id), domain, date(2016, 1, 1)))
        next_id += 1
    records.append(_record("900", "tie.co.uk", date(2017, 6, 1)))
    records.append(_record("901", "tie.co.uk", date(2017, 6, 1)))
    records.append(_record("902", "tie.co.uk", date(2019, 4, 1)))
    records.append(_record("903", "tie.co.uk", date(2019, 4, 1)))
    return _corpus(records)


def test_same_seed_returns_the_same_policy_ids(tmp_path: Path) -> None:
    corpus = _paired_corpus()
    sampling = _sampling()
    manifest = tmp_path / "gold.json"

    first = sample_gold_set(corpus, sampling, seed=42, manifest_path=manifest)
    second = sample_gold_set(corpus, sampling, seed=42)

    assert first.policy_ids == second.policy_ids
    assert len([item for item in first.selected if item.window == "pre"]) == 10
    assert len([item for item in first.selected if item.window == "post"]) == 10
    pre_domains = {item.domain for item in first.selected if item.window == "pre"}
    post_domains = {item.domain for item in first.selected if item.window == "post"}
    assert pre_domains == post_domains
    assert all(item.paired for item in first.selected)
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert written["role"] == "gold"
    assert written["policy_ids"] == first.policy_ids


def test_a_different_seed_can_change_the_draw() -> None:
    corpus = _paired_corpus()
    sampling = _sampling()

    first = sample_gold_set(corpus, sampling, seed=42)
    second = sample_gold_set(corpus, sampling, seed=7)

    assert first.policy_ids != second.policy_ids


def test_keeps_the_later_pre_snapshot_and_the_earlier_post_snapshot() -> None:
    records = [
        _record("1", "alpha.co.uk", date(2017, 1, 2)),
        _record("2", "alpha.co.uk", date(2017, 12, 1)),
        _record("3", "alpha.co.uk", date(2018, 6, 1)),
        _record("4", "alpha.co.uk", date(2019, 1, 15)),
        _record("5", "alpha.co.uk", date(2019, 12, 1)),
        _record("6", "alpha.co.uk", date(2016, 5, 1)),
    ]
    sample = sample_gold_set(_corpus(records), _sampling(), seed=42, per_window=1)
    selected = {item.window: item.snapshot_id for item in sample.selected}

    assert selected == {"pre": "2", "post": "4"}
    reasons = {item.snapshot_id: item.reason for item in sample.decisions}
    assert "later pre-window" in reasons["1"]
    assert "buffer" in reasons["3"]
    assert "earlier post-window" in reasons["5"]
    assert "outside" in reasons["6"]


def test_same_day_ties_use_the_snapshot_id() -> None:
    records = [
        _record("10", "alpha.co.uk", date(2017, 6, 1)),
        _record("11", "alpha.co.uk", date(2017, 6, 1)),
        _record("12", "alpha.co.uk", date(2019, 3, 1)),
        _record("13", "alpha.co.uk", date(2019, 3, 1)),
    ]
    sample = sample_gold_set(_corpus(records), _sampling(), seed=1, per_window=1)
    selected = {item.window: item.snapshot_id for item in sample.selected}

    assert selected == {"pre": "11", "post": "12"}


def test_unpaired_sites_fill_only_after_pairs_are_used() -> None:
    records = [
        _record("1", "paired.co.uk", date(2017, 2, 1)),
        _record("2", "paired.co.uk", date(2019, 2, 1)),
        _record("3", "preonly.co.uk", date(2017, 4, 1)),
        _record("4", "postonly.co.uk", date(2019, 4, 1)),
    ]
    sample = sample_gold_set(_corpus(records), _sampling(), seed=3, per_window=2)

    paired = [item for item in sample.selected if item.domain == "paired.co.uk"]
    assert len(paired) == 2
    assert all(item.paired for item in paired)
    assert any("fill the pre sample" in item.reason for item in sample.selected)
    assert any("fill the post sample" in item.reason for item in sample.selected)


def test_too_few_sites_raise() -> None:
    records = [_record("1", "only.co.uk", date(2017, 2, 1))]

    with pytest.raises(SamplingError, match="Need 10 pre"):
        sample_gold_set(_corpus(records), _sampling(), seed=42)


def test_inclusions_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    records = [
        _record("1", "alpha.co.uk", date(2017, 2, 1)),
        _record("2", "alpha.co.uk", date(2019, 2, 1)),
    ]
    logger = logging.getLogger("gdpr_gap_test_sampling")
    logger.setLevel(logging.INFO)

    with caplog.at_level(logging.INFO, logger=logger.name):
        sample_gold_set(_corpus(records), _sampling(), seed=42, per_window=1, logger=logger)

    included = [
        record for record in caplog.records if record.getMessage() == "gold_sample_included"
    ]
    assert any(record.getMessage() == "gold_sample" for record in caplog.records)
    assert {getattr(record, "fields")["domain"] for record in included} == {"alpha.co.uk"}

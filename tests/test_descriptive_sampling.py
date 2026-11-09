import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from gdpr_gap.config import SamplingConfig
from gdpr_gap.corpus import PolicyCorpus, PolicyRecord
from gdpr_gap.sampling import (
    DESCRIPTIVE_PER_WINDOW,
    NOT_VALIDATION,
    GoldSample,
    SamplingError,
    load_gold_manifest,
    sample_descriptive_set,
    sample_gold_set,
)


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


def _corpus(count: int) -> PolicyCorpus:
    records: list[PolicyRecord] = []
    next_id = 1
    for index in range(count):
        domain = f"site{index:02d}.co.uk"
        records.append(_record(str(next_id), domain, date(2017, 5, 1)))
        next_id += 1
        records.append(_record(str(next_id), domain, date(2019, 5, 1)))
        next_id += 1
    return PolicyCorpus(domain_suffix=".co.uk", records=records)


def test_default_size_is_40_pre_and_40_post(tmp_path: Path) -> None:
    manifest = tmp_path / "descriptive.json"
    sample = sample_descriptive_set(_corpus(45), _sampling(), seed=42, manifest_path=manifest)

    assert DESCRIPTIVE_PER_WINDOW == 40
    assert sample.role == "descriptive"
    assert sample.notice == NOT_VALIDATION
    assert sample.notice == "not validation unless labelled"
    pre = [item for item in sample.selected if item.window == "pre"]
    post = [item for item in sample.selected if item.window == "post"]
    assert len(pre) == 40
    assert len(post) == 40
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert written["notice"] == "not validation unless labelled"
    assert written["role"] == "descriptive"


def test_same_seed_is_stable_and_gold_sites_are_reserved() -> None:
    corpus = _corpus(8)
    sampling = _sampling()
    gold = sample_gold_set(corpus, sampling, seed=42, per_window=2)
    reserved = {item.domain for item in gold.selected}

    first = sample_descriptive_set(
        corpus, sampling, seed=42, per_window=2, reserved_domains=reserved
    )
    second = sample_descriptive_set(
        corpus, sampling, seed=42, per_window=2, reserved_domains=reserved
    )

    assert first.policy_ids == second.policy_ids
    assert set(first.policy_ids).isdisjoint(gold.policy_ids)
    reserved_reasons = [
        item.reason for item in first.decisions if item.domain in reserved and item.window == "pre"
    ]
    assert any("reserved for the gold set" in reason for reason in reserved_reasons)


def test_descriptive_manifest_cannot_be_loaded_as_gold(tmp_path: Path) -> None:
    manifest = tmp_path / "descriptive.json"
    sample = sample_descriptive_set(
        _corpus(3), _sampling(), seed=1, per_window=1, manifest_path=manifest
    )

    with pytest.raises(ValidationError):
        GoldSample.model_validate(sample.model_dump(mode="json"))
    with pytest.raises(SamplingError, match="not validation unless labelled"):
        load_gold_manifest(manifest)

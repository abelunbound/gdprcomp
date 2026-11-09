from pathlib import Path
from typing import Any

import pytest
import yaml

from gdpr_gap.criteria import CriteriaError, load_criteria, summary_table

CONSENT = (
    "The policy should detail procedures for obtaining, recording, "
    "and managing consent for data processing."
)


def test_all_22_criteria_load_and_render() -> None:
    catalog = load_criteria("configs/criteria.yaml")
    assert len(catalog.criteria) == 22
    assert [item.display_order for item in catalog.criteria] == list(range(1, 23))
    assert {item.weight for item in catalog.criteria} == {1}
    consent = next(item for item in catalog.criteria if item.id == "consent_mechanisms")
    assert consent.condition == CONSENT
    assert consent.legal_basis[0].article == "Article 4(11)"
    article_30 = [
        item.name for item in catalog.criteria if item.legal_basis[0].article == "Article 30"
    ]
    assert article_30 == ["Data Processing Records", "Records of Processing Activities"]
    assert "Article 30" in catalog.meta.duplicate_note

    table = summary_table(catalog)
    assert table.startswith("| Order | ID | Name | Weight | Legal basis |")
    body = [line for line in table.splitlines() if _is_data_row(line)]
    assert len(body) == 22
    for criterion in catalog.criteria:
        assert criterion.name in table
        assert criterion.id in table


def test_summary_table_follows_display_order(tmp_path: Path) -> None:
    catalog = load_criteria(_write(tmp_path, _payload()))
    swapped = catalog.model_copy(deep=True)
    swapped.criteria[0].display_order = 2
    swapped.criteria[1].display_order = 1
    table = summary_table(swapped)
    lines = [line for line in table.splitlines() if _is_data_row(line)]
    assert lines[0].startswith("| 1 | data_subject_rights |")
    assert lines[1].startswith("| 2 | data_processing_records |")


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["criteria"][1]["id"] = payload["criteria"][0]["id"]
    path = _write(tmp_path, payload)

    with pytest.raises(CriteriaError, match="duplicate id"):
        load_criteria(path)


def test_zero_weight_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["criteria"][0]["weight"] = 0
    path = _write(tmp_path, payload)

    with pytest.raises(CriteriaError, match="weight"):
        load_criteria(path)


def test_wrong_count_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["criteria"] = payload["criteria"][:-1]
    path = _write(tmp_path, payload)

    with pytest.raises(CriteriaError, match="expected 22 criteria, found 21"):
        load_criteria(path)


def _is_data_row(line: str) -> bool:
    return line.startswith("| ") and not line.startswith("| Order") and "---" not in line


def _payload() -> dict[str, Any]:
    loaded = yaml.safe_load(Path("configs/criteria.yaml").read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssertionError("criteria fixture must be a mapping")
    return loaded


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "criteria.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path

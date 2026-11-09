import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from gdpr_gap.calibration import (
    CalibrationError,
    LabelledSimilarity,
    apply_calibration,
    calibrate_retrieval_threshold,
    dev_policy_ids,
    load_calibration,
    record_calibration,
    save_calibration,
)
from gdpr_gap.config import load_config

KEY = "test-key-not-a-real-secret"


def _row(
    policy_id: str,
    similarity: float,
    verdict: Literal["Yes", "Somewhat", "No"],
) -> LabelledSimilarity:
    return LabelledSimilarity(
        policy_id=policy_id,
        criterion_id="consent_mechanisms",
        similarity=similarity,
        verdict=verdict,
    )


def test_dev_split_is_stable_and_holds_policies_out() -> None:
    first, held_first = dev_policy_ids(["b", "a", "c", "d"], 42)
    second, held_second = dev_policy_ids(["d", "c", "b", "a"], 42)

    assert (first, held_first) == (second, held_second)
    assert len(first) == 2
    assert set(first).isdisjoint(held_first)
    assert set(first + held_first) == {"a", "b", "c", "d"}


def test_threshold_is_fit_on_the_dev_split_and_ignores_held_out_rows() -> None:
    rows = [
        _row("dev", 0.9, "Yes"),
        _row("dev", 0.2, "No"),
        _row("held", 0.9, "No"),
    ]

    fitted = calibrate_retrieval_threshold(rows, seed=42, dev_ids=["dev"])

    assert fitted.threshold == 0.9
    assert fitted.metric_value == 1
    assert fitted.label_source == "model-produced"
    assert fitted.dev_policy_ids == ["dev"]
    assert fitted.held_out_policy_ids == ["held"]
    assert fitted.row_count == 2
    assert fitted.calibration_id == "retrieval_similarity"


def test_versions_are_kept_and_loaded_into_the_runtime_config(tmp_path: Path) -> None:
    directory = tmp_path / "calibration"
    original = calibrate_retrieval_threshold(
        [_row("dev", 0.9, "Yes"), _row("dev", 0.2, "No")],
        seed=42,
        dev_ids=["dev"],
    )
    changed = original.model_copy(update={"threshold": 0.4, "metric_value": 0.5})
    logger = logging.getLogger("gdpr_gap_calibration_test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger.addHandler(Capture())
    first = save_calibration(directory, original, logger=logger)
    again = save_calibration(directory, original)
    second = save_calibration(directory, changed)

    assert first.version == "v1"
    assert again.version == "v1"
    assert second.version == "v2"
    assert load_calibration(directory).version == "v2"
    assert load_calibration(directory, "v1").threshold == 0.9
    assert json.loads((directory / "v1.json").read_text(encoding="utf-8"))["threshold"] == 0.9
    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )
    applied = apply_calibration(config, load_calibration(directory, "v1"))
    assert config.thresholds.retrieval_similarity is None
    assert applied.thresholds.retrieval_similarity == 0.9
    fields = cast(dict[str, Any], getattr(records[0], "fields"))
    assert fields["version"] == "v1"
    assert fields["label_source"] == "model-produced"
    assert "dev" not in fields


def test_calibration_is_recorded_in_the_manifest(tmp_path: Path) -> None:
    fitted = calibrate_retrieval_threshold(
        [_row("dev", 0.9, "Somewhat"), _row("dev", 0.1, "No")],
        seed=1,
        dev_ids=["dev"],
    )
    stored = save_calibration(tmp_path, fitted)
    manifest = {"role": "gold", "seed": 1}

    stamped = record_calibration(manifest, stored)

    assert manifest == {"role": "gold", "seed": 1}
    assert stamped["calibration"] == {
        "calibration_id": "retrieval_similarity",
        "version": stored.version,
        "threshold": stored.threshold,
        "label_source": "model-produced",
    }
    assert load_calibration(tmp_path).threshold == stored.threshold


def test_dev_split_without_a_positive_label_fails() -> None:
    with pytest.raises(CalibrationError, match="Yes or Somewhat"):
        calibrate_retrieval_threshold([_row("dev", 0.2, "No")], seed=1, dev_ids=["dev"])

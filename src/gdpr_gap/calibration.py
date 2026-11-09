"""Calibrate the retrieval similarity threshold on a dev split.

The threshold is fitted from labelled similarities and stored as a versioned
file. Runtime code loads that file. It does not use a hardcoded cutoff.
Labels consumed here are model-produced, not expert labels and not ground truth.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gdpr_gap.config import AppConfig
from gdpr_gap.logging import log_event

_VERSION_FILE = re.compile(r"^v([1-9][0-9]*)\.json$")
_GRID_STEP = 0.01
_POSITIVE = frozenset({"Yes", "Somewhat"})


def _positive_verdicts() -> list[Literal["Somewhat", "Yes"]]:
    return ["Somewhat", "Yes"]


class CalibrationError(Exception):
    """Raised when a threshold cannot be fitted, stored, or loaded."""


class LabelledSimilarity(BaseModel):
    """One model-produced verdict and the best passage similarity for that criterion."""

    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    similarity: float = Field(ge=-1, le=1)
    verdict: Literal["Yes", "Somewhat", "No"]


class CalibrationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration_id: Literal["retrieval_similarity"] = "retrieval_similarity"
    version: str = Field(pattern=r"^v[1-9][0-9]*$")
    threshold: float = Field(ge=0, le=1)
    metric: Literal["binary_f1"] = "binary_f1"
    metric_value: float = Field(ge=0, le=1)
    grid_step: float = Field(default=0.01, gt=0, le=1)
    tie_break: Literal["highest threshold"] = "highest threshold"
    seed: int = Field(ge=0)
    label_source: Literal["model-produced"] = "model-produced"
    positive_verdicts: list[Literal["Somewhat", "Yes"]] = Field(default_factory=_positive_verdicts)
    dev_policy_ids: list[str] = Field(min_length=1)
    held_out_policy_ids: list[str]
    row_count: int = Field(ge=1)


def dev_policy_ids(policy_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Split policy ids into a dev half and a held-out half. The same seed is stable."""
    unique = sorted(set(policy_ids))
    if not unique:
        raise CalibrationError("Cannot split a dev set with no policies.")
    order = unique[:]
    random.Random(seed).shuffle(order)
    dev_count = (len(order) + 1) // 2
    return sorted(order[:dev_count]), sorted(order[dev_count:])


def calibrate_retrieval_threshold(
    rows: list[LabelledSimilarity],
    *,
    seed: int,
    dev_ids: list[str] | None = None,
) -> CalibrationRecord:
    """Fit retrieval_similarity on the dev policies. Held-out rows are not used."""
    if seed < 0:
        raise CalibrationError("Seed must be zero or greater.")
    if not rows:
        raise CalibrationError("Cannot calibrate with no labelled rows.")
    chosen, held_out = (
        (sorted(set(dev_ids)), sorted(set(row.policy_id for row in rows) - set(dev_ids)))
        if dev_ids is not None
        else dev_policy_ids([row.policy_id for row in rows], seed)
    )
    if not chosen:
        raise CalibrationError("Dev split has no policies.")
    dev_rows = [row for row in rows if row.policy_id in set(chosen)]
    if not dev_rows:
        raise CalibrationError("Dev split has no labelled rows.")
    if not any(row.verdict in _POSITIVE for row in dev_rows):
        raise CalibrationError("Dev split has no Yes or Somewhat label to calibrate against.")
    threshold, score = _best_threshold(dev_rows)
    return CalibrationRecord(
        version="v1",
        threshold=threshold,
        metric_value=score,
        seed=seed,
        dev_policy_ids=chosen,
        held_out_policy_ids=held_out,
        row_count=len(dev_rows),
    )


def save_calibration(
    directory: str | Path,
    record: CalibrationRecord,
    *,
    logger: logging.Logger | None = None,
) -> CalibrationRecord:
    """Store a fit. An identical latest fit keeps its version; a change writes the next one."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    found = versions(root)
    body = _body(record)
    if found:
        latest = load_calibration(root, found[-1])
        if _body(latest) == body:
            return latest
        version = f"v{_version_number(found[-1]) + 1}"
    else:
        version = "v1"
    stored = record.model_copy(update={"version": version})
    path = _file(root, version)
    if path.exists():
        raise CalibrationError(f"Refusing to overwrite calibration {version}: {path}.")
    _write(path, stored)
    if logger is not None:
        log_event(
            logger,
            "calibration_saved",
            calibration_id=stored.calibration_id,
            version=stored.version,
            threshold=stored.threshold,
            metric_value=stored.metric_value,
            dev_count=len(stored.dev_policy_ids),
            label_source=stored.label_source,
        )
    return stored


def load_calibration(directory: str | Path, version: str | None = None) -> CalibrationRecord:
    """Load one calibration version. Omit version to load the newest file."""
    root = Path(directory)
    chosen = version if version is not None else _latest(root)
    path = _file(root, chosen)
    if not path.is_file():
        raise CalibrationError(f"Calibration {chosen} was not found at {path}.")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"Calibration file is not JSON: {path}.") from exc
    if not isinstance(loaded, dict):
        raise CalibrationError(f"Calibration file must be an object: {path}.")
    return CalibrationRecord.model_validate(loaded)


def apply_calibration(config: AppConfig, record: CalibrationRecord) -> AppConfig:
    """Replace the configured retrieval threshold with the loaded calibration."""
    thresholds = config.thresholds.model_copy(update={"retrieval_similarity": record.threshold})
    return config.model_copy(update={"thresholds": thresholds})


def record_calibration(
    manifest: dict[str, object],
    record: CalibrationRecord,
) -> dict[str, object]:
    """Copy a manifest and record the calibration version that was loaded."""
    updated = dict(manifest)
    updated["calibration"] = {
        "calibration_id": record.calibration_id,
        "version": record.version,
        "threshold": record.threshold,
        "label_source": record.label_source,
    }
    return updated


def versions(directory: str | Path) -> list[str]:
    """Return calibration versions in numeric order."""
    root = Path(directory)
    if not root.is_dir():
        return []
    found: list[str] = []
    for path in root.iterdir():
        if _VERSION_FILE.match(path.name) is not None:
            found.append(path.stem)
    return sorted(found, key=_version_number)


def _best_threshold(rows: list[LabelledSimilarity]) -> tuple[float, float]:
    best_threshold = 0.0
    best_score = -1.0
    steps = int(round(1 / _GRID_STEP))
    for step in range(steps + 1):
        threshold = round(step * _GRID_STEP, 2)
        score = _f1(rows, threshold)
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


def _f1(rows: list[LabelledSimilarity], threshold: float) -> float:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for row in rows:
        predicted = row.similarity >= threshold
        actual = row.verdict in _POSITIVE
        if predicted and actual:
            true_positive += 1
        elif predicted and not actual:
            false_positive += 1
        elif actual:
            false_negative += 1
    if true_positive == 0:
        return 0.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return (2 * precision * recall) / (precision + recall)


def _body(record: CalibrationRecord) -> dict[str, object]:
    payload = record.model_dump(mode="json")
    del payload["version"]
    return payload


def _latest(directory: Path) -> str:
    found = versions(directory)
    if not found:
        raise CalibrationError(f"No calibration file was found in {directory}.")
    return found[-1]


def _file(directory: Path, version: str) -> Path:
    if _VERSION_FILE.match(f"{version}.json") is None:
        raise CalibrationError(f"Calibration version must look like v1, not {version!r}.")
    return directory / f"{version}.json"


def _write(path: Path, record: CalibrationRecord) -> None:
    payload = record.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _version_number(version: str) -> int:
    return int(version[1:])

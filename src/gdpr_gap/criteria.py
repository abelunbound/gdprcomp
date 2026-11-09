"""Load the 22 GDPR criteria from the single YAML source of truth."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

EXPECTED_CRITERIA = 22
DEFAULT_CRITERIA_PATH = Path("configs/criteria.yaml")


class CriteriaError(Exception):
    """Raised when the criteria file is missing or invalid."""


class LegalBasis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instrument: str = Field(min_length=1)
    article: str = Field(min_length=1)
    title: str = Field(min_length=1)


class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    weight: float = Field(gt=0)
    display_order: int = Field(ge=1)
    legal_basis: list[LegalBasis] = Field(min_length=1)
    related_provisions: list[str] = Field(min_length=1)
    note: str = Field(min_length=1)


class CriteriaMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    caveat: str = Field(min_length=1)
    duplicate_note: str = Field(min_length=1)


class CriteriaCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: CriteriaMeta
    criteria: list[Criterion]

    @model_validator(mode="after")
    def exactly_22_unique_criteria(self) -> CriteriaCatalog:
        count = len(self.criteria)
        if count != EXPECTED_CRITERIA:
            raise ValueError(f"expected {EXPECTED_CRITERIA} criteria, found {count}")
        self._reject_duplicates("id", [item.id for item in self.criteria])
        self._reject_duplicates("name", [item.name for item in self.criteria])
        orders = [item.display_order for item in self.criteria]
        self._reject_duplicates("display_order", orders)
        expected_orders = list(range(1, EXPECTED_CRITERIA + 1))
        if sorted(orders) != expected_orders:
            raise ValueError("display_order must be 1 through 22")
        return self

    @staticmethod
    def _reject_duplicates(field_name: str, values: Sequence[object]) -> None:
        seen: set[object] = set()
        for value in values:
            if value in seen:
                raise ValueError(f"duplicate {field_name}: {value}")
            seen.add(value)


def load_criteria(path: str | Path | None = None) -> CriteriaCatalog:
    """Load and validate the criteria catalog. No model call and no other config file."""
    criteria_path = Path(path) if path is not None else DEFAULT_CRITERIA_PATH
    if not criteria_path.is_file():
        raise CriteriaError(
            f"Criteria file not found: {criteria_path}. Expected {DEFAULT_CRITERIA_PATH}."
        )
    try:
        loaded = yaml.safe_load(criteria_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CriteriaError(f"Could not parse {criteria_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CriteriaError(
            f"Invalid criteria file {criteria_path}: the top level must be a mapping."
        )
    try:
        return CriteriaCatalog.model_validate(loaded)
    except ValidationError as exc:
        raise CriteriaError(_format_validation_error(criteria_path, exc)) from exc


def summary_table(catalog: CriteriaCatalog) -> str:
    """Render the catalog as a Markdown table. Uses only the loaded criteria."""
    header = "| Order | ID | Name | Weight | Legal basis |"
    separator = "| ---: | --- | --- | ---: | --- |"
    rows = [header, separator]
    for criterion in sorted(catalog.criteria, key=lambda item: item.display_order):
        basis = "; ".join(f"{item.instrument} {item.article}" for item in criterion.legal_basis)
        weight = f"{criterion.weight:g}"
        row = (
            f"| {criterion.display_order} | {criterion.id} | {criterion.name} "
            f"| {weight} | {basis} |"
        )
        rows.append(row)
    return "\n".join(rows)


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid criteria file {path}:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(file)"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)

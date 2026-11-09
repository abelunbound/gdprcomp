"""Compare whole-policy, hybrid, and Legal-BERT on the same labelled policies.

Each metric names the method with the better measured value. A method that was
not run, or that has no value for that metric, is left out of that choice.
Alternative embeddings are reported as unavailable. Hybrid is not treated as
the default winner.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gdpr_gap.metrics import LABEL_SOURCE, MetricsError, VerdictPair, classification_metrics

MethodName = Literal["whole_policy_llm", "hybrid", "legal_bert_only", "alternative_embeddings"]
ScoredMethod = Literal["whole_policy_llm", "hybrid", "legal_bert_only"]
MetricName = Literal["accuracy", "runtime", "cost", "repeat_consistency"]
_COMPARED: tuple[ScoredMethod, ...] = ("whole_policy_llm", "hybrid", "legal_bert_only")
_NAMES: dict[MethodName, str] = {
    "whole_policy_llm": "Whole-policy LLM",
    "hybrid": "Hybrid",
    "legal_bert_only": "Legal-BERT only",
    "alternative_embeddings": "Alternative embeddings",
}
_METRIC_NAMES: dict[MetricName, str] = {
    "accuracy": "Accuracy",
    "runtime": "Runtime",
    "cost": "Cost",
    "repeat_consistency": "Repeat consistency",
}


class ComparisonError(Exception):
    """Raised when methods cannot be compared on the labelled policies."""


class MethodResult(BaseModel):
    """One method's results on the labelled policies. Legal-BERT has no verdicts."""

    model_config = ConfigDict(extra="forbid")

    method: ScoredMethod
    predictions: list[VerdictPair] = Field(default_factory=list)
    runtime_seconds: float = Field(ge=0)
    cost_usd: float | None = None
    repeat_agreement: float | None = Field(default=None, ge=0, le=1)
    prediction_model_id: str | None = None

    @field_validator("cost_usd")
    @classmethod
    def cost_is_not_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("cost must be zero or greater")
        return value


class ComparedMethod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: MethodName
    status: Literal["scored", "not_run", "unavailable"]
    accuracy: float | None = None
    runtime_seconds: float | None = None
    cost_usd: float | None = None
    repeat_agreement: float | None = None
    note: str = ""


class MetricWinner(BaseModel):
    """The method or methods with the best measured value. A tie has no single winner."""

    model_config = ConfigDict(extra="forbid")

    metric: MetricName
    direction: Literal["higher", "lower"]
    winner: ScoredMethod | None = None
    tied: list[ScoredMethod] = Field(default_factory=list)
    value: float | None = None


class ComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_model_id: str = Field(min_length=1)
    label_source: Literal["model-produced"] = LABEL_SOURCE
    policy_count: int = Field(ge=1)
    pair_count: int = Field(ge=1)
    methods: list[ComparedMethod] = Field(min_length=4, max_length=4)
    winners: list[MetricWinner] = Field(min_length=4, max_length=4)


def compare_methods(
    methods: list[MethodResult],
    labels: list[VerdictPair],
    *,
    label_model_id: str,
) -> ComparisonReport:
    """Name the winner for accuracy, runtime, cost, and repeat consistency."""
    if not label_model_id:
        raise ComparisonError("The gold labeller model id is required.")
    if not methods:
        raise ComparisonError("Comparison needs at least one method.")
    _labels(labels)
    supplied = _by_method(methods)
    compared = [
        _compare_one(supplied.get(method), method, labels, label_model_id) for method in _COMPARED
    ]
    compared.append(
        ComparedMethod(
            method="alternative_embeddings",
            status="unavailable",
            note="not available",
        )
    )
    return ComparisonReport(
        label_model_id=label_model_id,
        policy_count=len({row.policy_id for row in labels}),
        pair_count=len(labels),
        methods=compared,
        winners=[
            _winner("accuracy", compared, higher=True),
            _winner("runtime", compared, higher=False),
            _winner("cost", compared, higher=False),
            _winner("repeat_consistency", compared, higher=True),
        ],
    )


def format_comparison(report: ComparisonReport) -> str:
    """Render one line per metric winner and one line for each method that was not scored."""
    lines = [f"Label source: {report.label_source}"]
    lines.extend(_winner_line(item) for item in report.winners)
    for method in report.methods:
        if method.status != "scored" or method.method == "legal_bert_only":
            lines.append(f"{_NAMES[method.method]}: {method.note}")
    return "\n".join(lines)


def export_comparison(report: ComparisonReport) -> str:
    """Export the comparison as JSON."""
    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _by_method(methods: list[MethodResult]) -> dict[ScoredMethod, MethodResult]:
    found: dict[ScoredMethod, MethodResult] = {}
    for item in methods:
        if item.method in found:
            raise ComparisonError(f"Method {item.method} is repeated.")
        found[item.method] = item
    return found


def _labels(labels: list[VerdictPair]) -> None:
    if not labels:
        raise ComparisonError("Comparison needs gold labels.")
    seen: set[tuple[str, str]] = set()
    for row in labels:
        key = (row.policy_id, row.criterion_id)
        if key in seen:
            raise ComparisonError(f"Duplicate label for {row.policy_id} {row.criterion_id}.")
        seen.add(key)


def _compare_one(
    item: MethodResult | None,
    method: ScoredMethod,
    labels: list[VerdictPair],
    label_model_id: str,
) -> ComparedMethod:
    if item is None:
        return ComparedMethod(method=method, status="not_run", note="not run")
    if item.prediction_model_id == label_model_id:
        raise ComparisonError("Predictions and labels come from the same system.")
    if method == "legal_bert_only":
        if item.predictions:
            raise ComparisonError("Legal-BERT only has no verdict to score.")
        return ComparedMethod(
            method=method,
            status="scored",
            runtime_seconds=item.runtime_seconds,
            cost_usd=item.cost_usd,
            repeat_agreement=item.repeat_agreement,
            note="retrieval signal, no verdict",
        )
    if item.prediction_model_id is None:
        raise ComparisonError(f"{_NAMES[method]} needs the judge model id.")
    try:
        classification_metrics(
            item.predictions,
            labels,
            prediction_model_id=item.prediction_model_id,
            label_model_id=label_model_id,
        )
    except MetricsError as exc:
        raise ComparisonError(str(exc)) from exc
    return ComparedMethod(
        method=method,
        status="scored",
        accuracy=_accuracy(item.predictions, labels),
        runtime_seconds=item.runtime_seconds,
        cost_usd=item.cost_usd,
        repeat_agreement=item.repeat_agreement,
    )


def _accuracy(predictions: list[VerdictPair], labels: list[VerdictPair]) -> float:
    predicted = {(row.policy_id, row.criterion_id): row.verdict for row in predictions}
    gold = {(row.policy_id, row.criterion_id): row.verdict for row in labels}
    matches = sum(predicted[key] == gold[key] for key in gold)
    return matches / len(gold)


def _winner(metric: MetricName, methods: list[ComparedMethod], *, higher: bool) -> MetricWinner:
    field = {
        "accuracy": "accuracy",
        "runtime": "runtime_seconds",
        "cost": "cost_usd",
        "repeat_consistency": "repeat_agreement",
    }[metric]
    scored: list[tuple[ScoredMethod, float]] = []
    for method in methods:
        if method.status != "scored" or method.method == "alternative_embeddings":
            continue
        value = getattr(method, field)
        if isinstance(value, float):
            scored.append((method.method, value))
    if not scored:
        return MetricWinner(metric=metric, direction="higher" if higher else "lower")
    values = [value for _method, value in scored]
    best = max(values) if higher else min(values)
    tied = sorted(method for method, value in scored if value == best)
    return MetricWinner(
        metric=metric,
        direction="higher" if higher else "lower",
        winner=tied[0] if len(tied) == 1 else None,
        tied=tied,
        value=best,
    )


def _winner_line(item: MetricWinner) -> str:
    name = _METRIC_NAMES[item.metric]
    if item.value is None:
        return f"{name}: no method reported a value"
    rendered = _value(item.metric, item.value)
    if item.winner is None:
        names = " and ".join(_NAMES[method] for method in item.tied)
        return f"{name}: tie between {names} ({rendered})"
    return f"{name}: {_NAMES[item.winner]} ({rendered})"


def _value(metric: MetricName, value: float) -> str:
    if metric == "runtime":
        return f"{value:.3f} seconds"
    if metric == "cost":
        return f"{value:.3f} USD"
    return f"{value:.3f}"

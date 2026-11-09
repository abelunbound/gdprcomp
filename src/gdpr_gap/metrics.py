"""Classification metrics for judge verdicts against gold labels.

Gap scores use is_gap. The three-class scores use Yes, Somewhat, and No.
Labels are model-produced. Predictions and labels from the same model are
rejected before any score is computed. Bootstrap intervals resample policies
with replacement and keep every judgement from a drawn policy together.
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gdpr_gap.gaps import is_gap

Verdict = Literal["Yes", "Somewhat", "No"]
ORDER: tuple[Verdict, ...] = ("No", "Somewhat", "Yes")
LABEL_SOURCE: Literal["model-produced"] = "model-produced"


class MetricsError(Exception):
    """Raised when classification metrics cannot be computed."""


class VerdictPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    verdict: Verdict


class ConfusionMatrix(BaseModel):
    """Rows are gold labels and columns are predictions, in No, Somewhat, Yes order."""

    model_config = ConfigDict(extra="forbid")

    labels: list[Verdict] = Field(default_factory=lambda: list(ORDER), min_length=3, max_length=3)
    counts: list[list[int]] = Field(min_length=3, max_length=3)


class GapScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)


class MetricBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_count: int = Field(ge=1)
    gap: GapScores
    macro_f1: float = Field(ge=0, le=1)
    weighted_kappa: float = Field(ge=-1, le=1)
    confusion: ConfusionMatrix


class CriterionMetrics(MetricBlock):
    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")


class ClassificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_model_id: str = Field(min_length=1)
    label_model_id: str = Field(min_length=1)
    label_source: Literal["model-produced"] = LABEL_SOURCE
    overall: MetricBlock
    criteria: list[CriterionMetrics] = Field(min_length=1)

    @model_validator(mode="after")
    def models_differ(self) -> ClassificationReport:
        if self.prediction_model_id == self.label_model_id:
            raise ValueError("predictions and labels come from the same system")
        return self


class StatisticInterval(BaseModel):
    """Point estimate and percentile interval. The estimate may sit outside the interval."""

    model_config = ConfigDict(extra="forbid")

    estimate: float
    lower: float
    upper: float


class BootstrapReport(BaseModel):
    """Policy-resampled intervals for gap F1, macro-F1, and weighted kappa."""

    model_config = ConfigDict(extra="forbid")

    prediction_model_id: str = Field(min_length=1)
    label_model_id: str = Field(min_length=1)
    label_source: Literal["model-produced"] = LABEL_SOURCE
    seed: int = Field(ge=0)
    resamples: int = Field(ge=1)
    level: float = Field(gt=0, lt=1)
    policy_count: int = Field(ge=2)
    gap_f1: StatisticInterval
    macro_f1: StatisticInterval
    weighted_kappa: StatisticInterval

    @model_validator(mode="after")
    def models_differ(self) -> BootstrapReport:
        if self.prediction_model_id == self.label_model_id:
            raise ValueError("predictions and labels come from the same system")
        return self


def classification_metrics(
    predictions: list[VerdictPair],
    labels: list[VerdictPair],
    *,
    prediction_model_id: str,
    label_model_id: str,
) -> ClassificationReport:
    """Score predictions against gold labels. The two model ids must differ."""
    if prediction_model_id == label_model_id:
        raise MetricsError("Predictions and labels come from the same system.")
    paired = _pair(predictions, labels)
    by_criterion: dict[str, list[tuple[Verdict, Verdict]]] = {}
    for _policy_id, criterion_id, gold, predicted in paired:
        by_criterion.setdefault(criterion_id, []).append((gold, predicted))
    criteria = [
        _criterion(criterion_id, rows) for criterion_id, rows in sorted(by_criterion.items())
    ]
    overall = _block([(gold, predicted) for _policy_id, _criterion, gold, predicted in paired])
    return ClassificationReport(
        prediction_model_id=prediction_model_id,
        label_model_id=label_model_id,
        overall=overall,
        criteria=criteria,
    )


def bootstrap_intervals(
    predictions: list[VerdictPair],
    labels: list[VerdictPair],
    *,
    prediction_model_id: str,
    label_model_id: str,
    seed: int,
    resamples: int = 1000,
    level: float = 0.95,
) -> BootstrapReport:
    """Resample policies with replacement and report percentile intervals."""
    scored = classification_metrics(
        predictions,
        labels,
        prediction_model_id=prediction_model_id,
        label_model_id=label_model_id,
    )
    _require_bootstrap(seed, resamples, level)
    grouped = _rows_by_policy(predictions, labels)
    if len(grouped) < 2:
        raise MetricsError("Bootstrap needs at least two policies.")
    draws = policy_draws(list(grouped), seed=seed, resamples=resamples)
    gap_f1: list[float] = []
    macro_f1: list[float] = []
    weighted_kappa: list[float] = []
    for draw in draws:
        block = _block(expand_policy_draw(draw, grouped))
        gap_f1.append(block.gap.f1)
        macro_f1.append(block.macro_f1)
        weighted_kappa.append(block.weighted_kappa)
    tail = (1 - level) / 2
    return BootstrapReport(
        prediction_model_id=prediction_model_id,
        label_model_id=label_model_id,
        seed=seed,
        resamples=resamples,
        level=level,
        policy_count=len(grouped),
        gap_f1=_interval(scored.overall.gap.f1, gap_f1, tail),
        macro_f1=_interval(scored.overall.macro_f1, macro_f1, tail),
        weighted_kappa=_interval(scored.overall.weighted_kappa, weighted_kappa, tail),
    )


def policy_draws(policy_ids: list[str], *, seed: int, resamples: int) -> list[tuple[str, ...]]:
    """Draw policy ids with replacement. Each draw has one copy per policy."""
    _require_bootstrap(seed, resamples, 0.95)
    if len(policy_ids) < 2:
        raise MetricsError("Bootstrap needs at least two policies.")
    generator = random.Random(seed)
    width = len(policy_ids)
    return [tuple(generator.choices(policy_ids, k=width)) for _index in range(resamples)]


def expand_policy_draw(
    draw: tuple[str, ...],
    rows_by_policy: dict[str, list[tuple[Verdict, Verdict]]],
) -> list[tuple[Verdict, Verdict]]:
    """Repeat every judgement of each drawn policy. Other policies are left out."""
    expanded: list[tuple[Verdict, Verdict]] = []
    for policy_id in draw:
        expanded.extend(rows_by_policy[policy_id])
    return expanded


def format_bootstrap(report: BootstrapReport) -> str:
    """Render the policy count and the three percentile intervals."""
    percent = round(report.level * 100)
    return "\n".join(
        [
            f"{report.policy_count} policies",
            (f"Gap F1 {percent}% interval: {report.gap_f1.lower:.3f} to {report.gap_f1.upper:.3f}"),
            (
                f"Macro-F1 {percent}% interval: "
                f"{report.macro_f1.lower:.3f} to {report.macro_f1.upper:.3f}"
            ),
            (
                f"Weighted kappa {percent}% interval: "
                f"{report.weighted_kappa.lower:.3f} to {report.weighted_kappa.upper:.3f}"
            ),
        ]
    )


def _require_bootstrap(seed: int, resamples: int, level: float) -> None:
    if seed < 0:
        raise MetricsError("Seed must be zero or greater.")
    if resamples < 1:
        raise MetricsError("Resamples must be at least 1.")
    if not 0 < level < 1:
        raise MetricsError("Level must be between 0 and 1.")


def _rows_by_policy(
    predictions: list[VerdictPair],
    labels: list[VerdictPair],
) -> dict[str, list[tuple[Verdict, Verdict]]]:
    grouped: dict[str, list[tuple[Verdict, Verdict]]] = {}
    for policy_id, _criterion_id, gold, predicted in _pair(predictions, labels):
        grouped.setdefault(policy_id, []).append((gold, predicted))
    return grouped


def _interval(estimate: float, samples: list[float], tail: float) -> StatisticInterval:
    return StatisticInterval(
        estimate=estimate,
        lower=_percentile(samples, tail),
        upper=_percentile(samples, 1 - tail),
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pair(
    predictions: list[VerdictPair],
    labels: list[VerdictPair],
) -> list[tuple[str, str, Verdict, Verdict]]:
    if not predictions or not labels:
        raise MetricsError("Metrics need both predictions and labels.")
    predicted = _index(predictions, "prediction")
    gold = _index(labels, "label")
    if set(predicted) != set(gold):
        raise MetricsError("Predictions and labels must cover the same policies and criteria.")
    paired: list[tuple[str, str, Verdict, Verdict]] = []
    for policy_id, criterion_id in sorted(gold):
        key = (policy_id, criterion_id)
        paired.append((policy_id, criterion_id, gold[key], predicted[key]))
    return paired


def _index(rows: list[VerdictPair], role: str) -> dict[tuple[str, str], Verdict]:
    found: dict[tuple[str, str], Verdict] = {}
    for row in rows:
        key = (row.policy_id, row.criterion_id)
        if key in found:
            raise MetricsError(f"Duplicate {role} for {row.policy_id} {row.criterion_id}.")
        found[key] = row.verdict
    return found


def _criterion(criterion_id: str, rows: list[tuple[Verdict, Verdict]]) -> CriterionMetrics:
    block = _block(rows)
    return CriterionMetrics(
        criterion_id=criterion_id,
        pair_count=block.pair_count,
        gap=block.gap,
        macro_f1=block.macro_f1,
        weighted_kappa=block.weighted_kappa,
        confusion=block.confusion,
    )


def _block(rows: list[tuple[Verdict, Verdict]]) -> MetricBlock:
    confusion = _confusion(rows)
    return MetricBlock(
        pair_count=len(rows),
        gap=_gap_scores(rows),
        macro_f1=_macro_f1(confusion.counts),
        weighted_kappa=_weighted_kappa(confusion.counts),
        confusion=confusion,
    )


def _gap_scores(rows: list[tuple[Verdict, Verdict]]) -> GapScores:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for gold, predicted in rows:
        gold_gap = is_gap(gold)
        predicted_gap = is_gap(predicted)
        if predicted_gap and gold_gap:
            true_positive += 1
        elif predicted_gap:
            false_positive += 1
        elif gold_gap:
            false_negative += 1
    precision, recall, f1 = _prf(true_positive, false_positive, false_negative)
    return GapScores(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def _confusion(rows: list[tuple[Verdict, Verdict]]) -> ConfusionMatrix:
    counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for gold, predicted in rows:
        counts[ORDER.index(gold)][ORDER.index(predicted)] += 1
    return ConfusionMatrix(counts=counts)


def _macro_f1(counts: list[list[int]]) -> float:
    scores: list[float] = []
    for index in range(3):
        true_positive = counts[index][index]
        false_positive = sum(counts[row][index] for row in range(3)) - true_positive
        false_negative = sum(counts[index]) - true_positive
        _precision, _recall, f1 = _prf(true_positive, false_positive, false_negative)
        scores.append(f1)
    return sum(scores) / 3


def _weighted_kappa(counts: list[list[int]]) -> float:
    total = sum(sum(row) for row in counts)
    observed = 0.0
    expected = 0.0
    row_totals = [sum(row) for row in counts]
    column_totals = [sum(counts[row][column] for row in range(3)) for column in range(3)]
    for gold_index in range(3):
        for predicted_index in range(3):
            weight = 1 - abs(gold_index - predicted_index) / 2
            observed += weight * counts[gold_index][predicted_index] / total
            chance = (row_totals[gold_index] / total) * (column_totals[predicted_index] / total)
            expected += weight * chance
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def _prf(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> tuple[float, float, float]:
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel


class EvaluationObservation(Protocol):
    task_id: str
    category: str
    passed: bool
    final_probability: float
    latency_ms: float
    cost_usd: float


class ReliabilitySummary(BaseModel):
    task_id: str
    attempts: int
    successes: int
    pass_rate: float
    pass_at_k: bool
    pass_to_k: bool
    mean_probability: float
    probability_variance: float
    confidence_low: float
    confidence_high: float


class CategorySummary(BaseModel):
    category: str
    attempts: int
    successes: int
    pass_rate: float
    mean_probability: float
    mean_latency_ms: float
    total_cost_usd: float


def wilson_interval(successes: int, attempts: int, z: float = 1.96) -> tuple[float, float]:
    if attempts == 0:
        return 0.0, 0.0
    proportion = successes / attempts
    denominator = 1 + z * z / attempts
    centre = proportion + z * z / (2 * attempts)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / attempts + z * z / (4 * attempts * attempts)
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def summarize_reliability(
    results: Sequence[EvaluationObservation],
) -> list[ReliabilitySummary]:
    grouped: dict[str, list[EvaluationObservation]] = defaultdict(list)
    for result in results:
        grouped[result.task_id].append(result)
    summaries: list[ReliabilitySummary] = []
    for task_id, attempts in sorted(grouped.items()):
        successes = sum(item.passed for item in attempts)
        probabilities = [item.final_probability for item in attempts]
        low, high = wilson_interval(successes, len(attempts))
        summaries.append(
            ReliabilitySummary(
                task_id=task_id,
                attempts=len(attempts),
                successes=successes,
                pass_rate=successes / len(attempts),
                pass_at_k=successes > 0,
                pass_to_k=successes == len(attempts),
                mean_probability=sum(probabilities) / len(probabilities),
                probability_variance=_variance(probabilities),
                confidence_low=low,
                confidence_high=high,
            )
        )
    return summaries


def summarize_categories(
    results: Sequence[EvaluationObservation],
) -> list[CategorySummary]:
    grouped: dict[str, list[EvaluationObservation]] = defaultdict(list)
    for result in results:
        grouped[result.category].append(result)
    summaries: list[CategorySummary] = []
    for category, attempts in sorted(grouped.items()):
        successes = sum(item.passed for item in attempts)
        summaries.append(
            CategorySummary(
                category=category,
                attempts=len(attempts),
                successes=successes,
                pass_rate=successes / len(attempts),
                mean_probability=sum(item.final_probability for item in attempts) / len(attempts),
                mean_latency_ms=sum(item.latency_ms for item in attempts) / len(attempts),
                total_cost_usd=sum(item.cost_usd for item in attempts),
            )
        )
    return summaries
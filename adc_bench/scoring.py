from __future__ import annotations

from collections import defaultdict
from typing import Any


ADC_WEIGHTS = {
    "correctness": 0.35,
    "complexity": 0.20,
    "trace_quality": 0.15,
    "transfer": 0.15,
    "robustness": 0.10,
    "anti_cheat": 0.05,
}


def adc_score(result: dict[str, Any]) -> float:
    score = 0.0
    for field, weight in ADC_WEIGHTS.items():
        score += weight * float(result.get(field, 0.0))
    return round(score, 6)


def apply_transfer_scores(results: list[dict[str, Any]], metadata_by_task: dict[str, Any]) -> None:
    family_scores: dict[str, list[float]] = defaultdict(list)
    for result in results:
        metadata = metadata_by_task[result["task_id"]]
        family_scores[metadata.family].append(float(result.get("correctness", 0.0)))

    family_average = {
        family: (sum(scores) / len(scores) if scores else 0.0)
        for family, scores in family_scores.items()
    }
    for result in results:
        metadata = metadata_by_task[result["task_id"]]
        result["transfer"] = round(family_average.get(metadata.family, 0.0), 6)
        result["adc_score"] = adc_score(result)

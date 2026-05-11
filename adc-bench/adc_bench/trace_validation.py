from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schemas import TraceValidationResult, load_json


TRACE_WEIGHTS = {
    "chosen_algorithm": 0.35,
    "invariant": 0.25,
    "complexity_time": 0.15,
    "rejected_algorithms": 0.15,
    "edge_cases": 0.10,
}


def validate_trace(trace_path: Path | None, expected_trace_path: Path) -> TraceValidationResult:
    if trace_path is None or not trace_path.exists():
        return TraceValidationResult(
            score=0.0,
            details={key: 0.0 for key in TRACE_WEIGHTS},
            errors=["trace.json not found"],
        )

    errors: list[str] = []
    try:
        trace = load_json(trace_path)
    except Exception as exc:  # pragma: no cover - exercised by CLI users.
        return TraceValidationResult(
            score=0.0,
            details={key: 0.0 for key in TRACE_WEIGHTS},
            errors=[str(exc)],
        )

    try:
        expected = load_json(expected_trace_path)
    except Exception as exc:
        return TraceValidationResult(
            score=0.0,
            details={key: 0.0 for key in TRACE_WEIGHTS},
            errors=[str(exc)],
        )

    _validate_trace_shape(trace, errors)
    details = {
        "chosen_algorithm": _score_text_field(
            trace.get("chosen_algorithm", ""),
            expected.get("target_keywords", []),
            TRACE_WEIGHTS["chosen_algorithm"],
        ),
        "invariant": _score_text_field(
            trace.get("invariant", ""),
            expected.get("invariant_keywords", []),
            TRACE_WEIGHTS["invariant"],
        ),
        "complexity_time": _score_text_field(
            _normalize_complexity(trace.get("complexity_time", "")),
            [_normalize_complexity(item) for item in expected.get("complexity_time_keywords", [])],
            TRACE_WEIGHTS["complexity_time"],
        ),
        "rejected_algorithms": _score_rejected_algorithms(
            trace.get("rejected_algorithms", []),
            expected.get("rejected_keywords", []),
            TRACE_WEIGHTS["rejected_algorithms"],
        ),
        "edge_cases": _score_edge_cases(
            trace.get("edge_cases", []),
            expected.get("edge_case_keywords", []),
            TRACE_WEIGHTS["edge_cases"],
        ),
    }
    score = min(1.0, round(sum(details.values()), 6))
    return TraceValidationResult(score=score, details=details, errors=errors)


def _validate_trace_shape(trace: dict[str, Any], errors: list[str]) -> None:
    required = {
        "chosen_algorithm": str,
        "hypotheses": list,
        "rejected_algorithms": list,
        "invariant": str,
        "complexity_time": str,
        "complexity_space": str,
        "edge_cases": list,
    }
    for field, expected_type in required.items():
        if field not in trace:
            errors.append(f"trace.json missing field {field!r}")
        elif not isinstance(trace[field], expected_type):
            errors.append(f"trace.json field {field!r} should be {expected_type.__name__}")


def _score_text_field(text: Any, keywords: list[str], weight: float) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    if not keywords:
        return weight
    normalized = _normalize_text(text)
    matches = sum(1 for keyword in keywords if _normalize_text(keyword) in normalized)
    if matches == 0:
        return 0.0
    return round(weight * min(1.0, matches / max(1, min(3, len(keywords)))), 6)


def _score_rejected_algorithms(rejected: Any, keywords: list[str], weight: float) -> float:
    if not isinstance(rejected, list) or not rejected:
        return 0.0
    text_parts: list[str] = []
    for item in rejected:
        if isinstance(item, dict):
            text_parts.append(str(item.get("name", "")))
            text_parts.append(str(item.get("reason", "")))
        else:
            text_parts.append(str(item))
    combined = _normalize_text(" ".join(text_parts))
    if not keywords:
        return weight
    matches = sum(1 for keyword in keywords if _normalize_text(keyword) in combined)
    if matches == 0:
        return 0.0
    return round(weight * min(1.0, matches / max(1, min(2, len(keywords)))), 6)


def _score_edge_cases(edge_cases: Any, keywords: list[str], weight: float) -> float:
    if not isinstance(edge_cases, list) or not edge_cases:
        return 0.0
    combined = _normalize_text(" ".join(str(item) for item in edge_cases))
    if not keywords:
        return weight
    matches = sum(1 for keyword in keywords if _normalize_text(keyword) in combined)
    if matches == 0:
        return round(weight * 0.4, 6)
    return round(weight * min(1.0, matches / max(1, min(2, len(keywords)))), 6)


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _normalize_complexity(text: Any) -> str:
    normalized = _normalize_text(text)
    replacements = {
        "linearithmic": "o(n log n)",
        "linear": "o(n)",
        "quadratic": "o(n^2)",
        "constant": "o(1)",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized.replace(" ", "")

from __future__ import annotations

import re
from pathlib import Path

from .schemas import AntiCheatResult


SUSPICIOUS_STRINGS = [
    "hidden_tests",
    "expected_trace",
    "reference_solution",
    "open(",
    "os.listdir",
    "__file__",
    "inspect",
    "subprocess",
    "eval(",
    "exec(",
]

ALLOWED_SUBMISSION_FILES = {"solution.py", "trace.json", "README.md"}


def scan_submission(solution_path: Path, submission_dir: Path | None = None) -> AntiCheatResult:
    violations: list[str] = []
    try:
        source = solution_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return AntiCheatResult(False, 0.0, [f"solution.py not found at {solution_path}"])

    lowered = source.lower()
    for needle in SUSPICIOUS_STRINGS:
        if needle.lower() in lowered:
            violations.append(f"solution.py contains suspicious string {needle!r}")

    if _looks_like_hardcoded_table(source):
        violations.append("solution.py appears to contain a hardcoded input/output table")

    if submission_dir is not None and submission_dir.exists():
        for child in submission_dir.iterdir():
            if child.is_file() and child.name not in ALLOWED_SUBMISSION_FILES:
                violations.append(f"unexpected submission file {child.name!r}")

    passed = not violations
    return AntiCheatResult(passed=passed, score=1.0 if passed else 0.0, violations=violations)


def _looks_like_hardcoded_table(source: str) -> bool:
    lowered = source.lower()
    table_name = re.search(r"\b(answers|outputs|known_cases|case_table|lookup)\s*=", lowered)
    many_literal_pairs = len(re.findall(r"[\{\[]\s*[\(\[]", source)) >= 3
    many_returns = len(re.findall(r"\breturn\s+[\{\[]", source)) >= 2
    return bool(table_name and (many_literal_pairs or many_returns))

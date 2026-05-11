from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from .anti_cheat import scan_submission
from .report import write_reports
from .sandbox import run_test_suite
from .schemas import TaskMetadata
from .scoring import adc_score, apply_transfer_scores
from .trace_validation import validate_trace


TEST_SUITES = ("public_tests.py", "hidden_tests.py", "stress_tests.py")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_tasks(tasks_root: Path | None = None) -> list[Path]:
    root = tasks_root or project_root() / "tasks"
    if not root.exists():
        raise FileNotFoundError(f"tasks directory not found: {root}")
    return sorted(path.parent for path in root.rglob("metadata.json"))


def load_metadata_by_task(tasks: list[Path]) -> dict[str, TaskMetadata]:
    metadata_by_task: dict[str, TaskMetadata] = {}
    for task in tasks:
        metadata = TaskMetadata.from_file(task / "metadata.json")
        metadata_by_task[metadata.task_id] = metadata
    return metadata_by_task


def run_task(task_dir: Path, submission_dir: Path | None = None, timeout_seconds: float = 3.0) -> dict[str, Any]:
    task_dir = task_dir.resolve()
    metadata = TaskMetadata.from_file(task_dir / "metadata.json")
    submission_dir = submission_dir.resolve() if submission_dir else None
    trace_path = (submission_dir / "trace.json") if submission_dir and (submission_dir / "trace.json").exists() else None
    errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="adc_submission_") as temp_name:
        temp_dir = Path(temp_name)
        solution_path = _materialize_solution(task_dir, submission_dir, temp_dir)
        anti_cheat = scan_submission(solution_path, submission_dir if submission_dir and submission_dir.exists() else None)
        suite_results = {}
        if anti_cheat.passed:
            for suite in TEST_SUITES:
                suite_results[suite] = run_test_suite(
                    suite_name=suite,
                    task_dir=task_dir,
                    solution_path=solution_path,
                    timeout_seconds=timeout_seconds,
                )
        else:
            errors.extend(anti_cheat.violations)

    for result in suite_results.values():
        errors.extend(f"{result.suite}: {error}" for error in result.errors[:3])

    public = suite_results.get("public_tests.py")
    hidden = suite_results.get("hidden_tests.py")
    stress = suite_results.get("stress_tests.py")
    hidden_ratio = hidden.pass_ratio if hidden else 0.0
    stress_passed = bool(stress and stress.passed and not stress.timed_out)
    trace_result = validate_trace(trace_path, task_dir / "expected_trace.json")
    errors.extend(f"trace: {error}" for error in trace_result.errors)

    robustness = hidden_ratio
    if metadata.split == "adversarial":
        robustness = hidden_ratio

    result = {
        "task_id": metadata.task_id,
        "correctness": round(hidden_ratio, 6),
        "complexity": 1.0 if stress_passed else 0.0,
        "trace_quality": trace_result.score,
        "transfer": 0.0,
        "robustness": round(robustness, 6),
        "anti_cheat": anti_cheat.score,
        "adc_score": 0.0,
        "passed_public": bool(public and public.passed),
        "passed_hidden": bool(hidden and hidden.passed),
        "passed_stress": stress_passed,
        "errors": errors,
    }
    result["adc_score"] = adc_score(result)
    return result


def run_all(
    submissions_root: Path,
    tasks_root: Path | None = None,
    reports_root: Path | None = None,
    timeout_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    tasks = discover_tasks(tasks_root)
    metadata_by_task = load_metadata_by_task(tasks)
    results: list[dict[str, Any]] = []
    for task in tasks:
        submission = _find_submission_for_task(submissions_root, task)
        results.append(run_task(task, submission, timeout_seconds=timeout_seconds))
    apply_transfer_scores(results, metadata_by_task)
    write_reports(results, reports_root or project_root() / "reports")
    return results


def _materialize_solution(task_dir: Path, submission_dir: Path | None, temp_dir: Path) -> Path:
    temp_solution = temp_dir / "solution.py"
    submitted = submission_dir / "solution.py" if submission_dir else None
    if submitted and submitted.exists():
        shutil.copy2(submitted, temp_solution)
    else:
        shutil.copy2(task_dir / "starter.py", temp_solution)
    return temp_solution


def _find_submission_for_task(submissions_root: Path, task_dir: Path) -> Path:
    metadata = TaskMetadata.from_file(task_dir / "metadata.json")
    candidates = [
        submissions_root / task_dir.name,
        submissions_root / metadata.task_id,
        submissions_root / metadata.task_id.replace("/", "__"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

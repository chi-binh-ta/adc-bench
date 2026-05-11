from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from .evaluator import project_root, run_task
from .schemas import TaskMetadata


AGENT_SUSPICIOUS_STRINGS = [
    "hidden_tests",
    "stress_tests",
    "reference_solution",
    "expected_trace",
    "os.listdir",
    "os.walk",
    "glob(",
    "__file__",
    "inspect",
    "subprocess",
    "eval(",
    "exec(",
]

SELF_IMPROVE_WEIGHTS = {
    "normalized_gain": 0.50,
    "no_regression": 0.20,
    "improvement_trace_quality": 0.15,
    "anti_cheat": 0.10,
    "reproducibility": 0.05,
}


@dataclass
class SelfImproveResult:
    task_dir: str
    baseline_agent_dir: str
    candidate_agent_dir: str
    train_tasks: list[str]
    heldout_tasks: list[str]
    baseline_results: dict[str, dict[str, Any]]
    candidate_results: dict[str, dict[str, Any]]
    baseline_average_score: float
    candidate_average_score: float
    absolute_gain: float
    normalized_gain: float
    no_regression: float
    anti_cheat: float
    anti_cheat_violations: list[str]
    improvement_trace_quality: float
    improvement_trace_errors: list[str]
    reproducibility: float
    self_improve_score: float
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_task_list(path: Path) -> list[Path]:
    """Load a JSON array of task paths, resolving relative paths from repo root."""
    root = project_root()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{path} must contain a JSON array of task path strings")

    tasks: list[Path] = []
    for item in raw:
        item_path = Path(item)
        if item_path.is_absolute():
            resolved = item_path
        else:
            resolved = root / item_path
            if not resolved.exists():
                resolved = path.parent / item_path
        if not resolved.exists():
            raise FileNotFoundError(f"Task path from {path} does not exist: {item}")
        tasks.append(resolved.resolve())
    return tasks


def run_agent_on_tasks(
    agent_dir: Path,
    tasks: list[Path],
    work_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    agent_dir = agent_dir.resolve()
    agent_path = agent_dir / "agent.py"
    work_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    if not agent_path.exists():
        message = f"Agent entrypoint not found: {agent_path}"
        errors.append(message)
        for task in tasks:
            task_id = _task_id(task)
            results[task_id] = _failed_task_result(task_id, [message])
        return {
            "agent_dir": str(agent_dir),
            "work_root": str(work_root),
            "results": results,
            "errors": errors,
        }

    for task in tasks:
        task_id = _task_id(task)
        output_dir = work_root / _safe_task_name(task_id)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(agent_path),
            "--task",
            str(task.resolve()),
            "--out",
            str(output_dir.resolve()),
        ]
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=project_root(),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
            return_code = completed.returncode
            stdout = _snippet(_sanitize_work_path(completed.stdout, work_root))
            stderr = _snippet(_sanitize_work_path(completed.stderr, work_root))
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            return_code = -1
            stdout = _snippet(_sanitize_work_path(exc.stdout or "", work_root))
            stderr = _snippet(_sanitize_work_path(exc.stderr or "", work_root))

        agent_errors: list[str] = []
        if timed_out:
            agent_errors.append(f"agent timed out after {timeout_seconds:.1f}s")
        elif return_code != 0:
            agent_errors.append(f"agent exited with status {return_code}")

        if not (output_dir / "solution.py").exists():
            agent_errors.append("agent did not produce solution.py")
        if not (output_dir / "trace.json").exists():
            agent_errors.append("agent did not produce trace.json")

        if agent_errors:
            result = _failed_task_result(task_id, agent_errors)
        else:
            result = run_task(task, output_dir, timeout_seconds=timeout_seconds)

        result["agent_returncode"] = return_code
        result["agent_stdout"] = stdout
        result["agent_stderr"] = stderr
        result["submission_dir"] = _display_work_path(output_dir, work_root)
        results[task_id] = result

    return {
        "agent_dir": str(agent_dir),
        "work_root": "<agent_work>",
        "results": results,
        "errors": errors,
    }


def evaluate_self_improvement(
    baseline_agent_dir: Path,
    candidate_agent_dir: Path,
    self_improve_task_dir: Path,
    timeout_seconds: float = 5.0,
) -> SelfImproveResult:
    task_dir = self_improve_task_dir.resolve()
    baseline_agent_dir = baseline_agent_dir.resolve()
    candidate_agent_dir = candidate_agent_dir.resolve()
    errors: list[str] = []

    try:
        train_tasks = load_task_list(task_dir / "train_tasks.json")
        heldout_tasks = load_task_list(task_dir / "heldout_tasks.json")
    except Exception as exc:
        return _empty_result(
            task_dir=task_dir,
            baseline_agent_dir=baseline_agent_dir,
            candidate_agent_dir=candidate_agent_dir,
            errors=[str(exc)],
        )

    anti_cheat, anti_cheat_violations = _scan_agent(candidate_agent_dir / "agent.py")
    improvement_trace_quality, improvement_trace_errors = _score_improvement_trace(
        candidate_agent_dir / "improvement_trace.json",
        task_dir / "expected_improvement_trace.json",
    )

    with tempfile.TemporaryDirectory(prefix="adc_self_improve_") as tmp:
        work_root = Path(tmp)
        baseline_run = run_agent_on_tasks(
            baseline_agent_dir,
            heldout_tasks,
            work_root / "baseline",
            timeout_seconds,
        )
        candidate_run = run_agent_on_tasks(
            candidate_agent_dir,
            heldout_tasks,
            work_root / "candidate",
            timeout_seconds,
        )

    errors.extend(baseline_run.get("errors", []))
    errors.extend(candidate_run.get("errors", []))

    baseline_results = baseline_run["results"]
    candidate_results = candidate_run["results"]
    baseline_average = _average_adc_score(baseline_results)
    candidate_average = _average_adc_score(candidate_results)
    absolute_gain = round(candidate_average - baseline_average, 6)
    normalized_gain = round(max(0.0, absolute_gain), 6)
    no_regression = _no_regression(baseline_results, candidate_results)
    reproducibility = 1.0
    self_improve_score = _self_improve_score(
        normalized_gain=normalized_gain,
        no_regression=no_regression,
        improvement_trace_quality=improvement_trace_quality,
        anti_cheat=anti_cheat,
        reproducibility=reproducibility,
    )

    return SelfImproveResult(
        task_dir=str(task_dir),
        baseline_agent_dir=str(baseline_agent_dir),
        candidate_agent_dir=str(candidate_agent_dir),
        train_tasks=[_display_task(task) for task in train_tasks],
        heldout_tasks=[_display_task(task) for task in heldout_tasks],
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        baseline_average_score=baseline_average,
        candidate_average_score=candidate_average,
        absolute_gain=absolute_gain,
        normalized_gain=normalized_gain,
        no_regression=no_regression,
        anti_cheat=anti_cheat,
        anti_cheat_violations=anti_cheat_violations,
        improvement_trace_quality=improvement_trace_quality,
        improvement_trace_errors=improvement_trace_errors,
        reproducibility=reproducibility,
        self_improve_score=self_improve_score,
        errors=errors,
    )


def format_self_improve_text_report(result: SelfImproveResult) -> str:
    lines = [
        "ADC-bench Self-Improvement Report",
        "=================================",
        f"Task: {_relativize(Path(result.task_dir))}",
        f"Baseline agent: {_relativize(Path(result.baseline_agent_dir))}",
        f"Candidate agent: {_relativize(Path(result.candidate_agent_dir))}",
        "",
        "Scores",
        "------",
        f"baseline_average_score   : {result.baseline_average_score:.3f}",
        f"candidate_average_score  : {result.candidate_average_score:.3f}",
        f"absolute_gain            : {result.absolute_gain:.3f}",
        f"normalized_gain          : {result.normalized_gain:.3f}",
        f"no_regression            : {result.no_regression:.3f}",
        f"improvement_trace_quality: {result.improvement_trace_quality:.3f}",
        f"anti_cheat               : {result.anti_cheat:.3f}",
        f"reproducibility          : {result.reproducibility:.3f}",
        f"self_improve_score       : {result.self_improve_score:.3f}",
        "",
        "Held-out Tasks",
        "--------------",
    ]

    for task_id in sorted(result.candidate_results):
        baseline_score = result.baseline_results.get(task_id, {}).get("adc_score", 0.0)
        candidate_score = result.candidate_results.get(task_id, {}).get("adc_score", 0.0)
        marker = "ok" if candidate_score >= baseline_score else "regression"
        lines.append(
            f"- {task_id}: baseline={baseline_score:.3f} "
            f"candidate={candidate_score:.3f} ({marker})"
        )

    lines.extend(["", "Warnings / Errors", "-----------------"])
    warnings = (
        result.errors
        + [f"anti-cheat: {item}" for item in result.anti_cheat_violations]
        + [f"improvement trace: {item}" for item in result.improvement_trace_errors]
    )
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("None")

    return "\n".join(lines)


def _score_improvement_trace(trace_path: Path, expected_path: Path) -> tuple[float, list[str]]:
    if not trace_path.exists():
        return 0.0, [f"improvement_trace.json not found at {trace_path}"]
    if not expected_path.exists():
        return 0.0, [f"expected improvement trace not found at {expected_path}"]

    errors: list[str] = []
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8-sig"))
        expected = json.loads(expected_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return 0.0, [f"invalid improvement trace JSON: {exc}"]

    if not isinstance(trace, dict) or not isinstance(expected, dict):
        return 0.0, ["improvement trace and expected trace must be JSON objects"]

    text = json.dumps(trace, ensure_ascii=False).lower()
    buckets = expected.get("keyword_buckets", {})
    if not isinstance(buckets, dict) or not buckets:
        return 0.0, ["expected improvement trace has no keyword_buckets"]

    matched = 0
    for bucket, keywords in buckets.items():
        if not isinstance(keywords, list):
            errors.append(f"keyword bucket {bucket!r} is not a list")
            continue
        normalized = [str(keyword).lower() for keyword in keywords]
        if any(keyword in text for keyword in normalized):
            matched += 1
        else:
            errors.append(f"missing improvement trace evidence for {bucket}")

    return round(matched / len(buckets), 6), errors


def _scan_agent(agent_path: Path) -> tuple[float, list[str]]:
    if not agent_path.exists():
        return 0.0, [f"candidate agent not found at {agent_path}"]

    source = agent_path.read_text(encoding="utf-8", errors="ignore").lower()
    violations = [
        f"candidate agent contains suspicious string {token!r}"
        for token in AGENT_SUSPICIOUS_STRINGS
        if token.lower() in source
    ]
    return (0.0 if violations else 1.0), violations


def _self_improve_score(
    *,
    normalized_gain: float,
    no_regression: float,
    improvement_trace_quality: float,
    anti_cheat: float,
    reproducibility: float,
) -> float:
    score = (
        SELF_IMPROVE_WEIGHTS["normalized_gain"] * normalized_gain
        + SELF_IMPROVE_WEIGHTS["no_regression"] * no_regression
        + SELF_IMPROVE_WEIGHTS["improvement_trace_quality"] * improvement_trace_quality
        + SELF_IMPROVE_WEIGHTS["anti_cheat"] * anti_cheat
        + SELF_IMPROVE_WEIGHTS["reproducibility"] * reproducibility
    )
    return round(score, 6)


def _average_adc_score(results: dict[str, dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return round(
        sum(float(item.get("adc_score", 0.0)) for item in results.values()) / len(results),
        6,
    )


def _no_regression(
    baseline_results: dict[str, dict[str, Any]],
    candidate_results: dict[str, dict[str, Any]],
) -> float:
    if not baseline_results:
        return 0.0
    ok = 0
    for task_id, baseline in baseline_results.items():
        baseline_score = float(baseline.get("adc_score", 0.0))
        candidate_score = float(candidate_results.get(task_id, {}).get("adc_score", 0.0))
        if candidate_score >= baseline_score:
            ok += 1
    return round(ok / len(baseline_results), 6)


def _failed_task_result(task_id: str, errors: list[str]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "correctness": 0.0,
        "complexity": 0.0,
        "trace_quality": 0.0,
        "transfer": 0.0,
        "robustness": 0.0,
        "anti_cheat": 0.0,
        "adc_score": 0.0,
        "passed_public": False,
        "passed_hidden": False,
        "passed_stress": False,
        "errors": errors,
    }


def _empty_result(
    *,
    task_dir: Path,
    baseline_agent_dir: Path,
    candidate_agent_dir: Path,
    errors: list[str],
) -> SelfImproveResult:
    return SelfImproveResult(
        task_dir=str(task_dir),
        baseline_agent_dir=str(baseline_agent_dir),
        candidate_agent_dir=str(candidate_agent_dir),
        train_tasks=[],
        heldout_tasks=[],
        baseline_results={},
        candidate_results={},
        baseline_average_score=0.0,
        candidate_average_score=0.0,
        absolute_gain=0.0,
        normalized_gain=0.0,
        no_regression=0.0,
        anti_cheat=0.0,
        anti_cheat_violations=[],
        improvement_trace_quality=0.0,
        improvement_trace_errors=[],
        reproducibility=0.0,
        self_improve_score=0.0,
        errors=errors,
    )


def _task_id(task_dir: Path) -> str:
    try:
        return TaskMetadata.from_file(task_dir / "metadata.json").task_id
    except Exception:
        return task_dir.name


def _safe_task_name(task_id: str) -> str:
    return task_id.replace("/", "__").replace("\\", "__")


def _snippet(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... truncated ..."


def _sanitize_work_path(text: str | bytes, work_root: Path) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if not text:
        return ""
    variants = {str(work_root), str(work_root).replace("\\", "\\\\")}
    sanitized = text
    for variant in variants:
        sanitized = sanitized.replace(variant, "<agent_work>")
    return sanitized


def _display_work_path(path: Path, work_root: Path) -> str:
    try:
        return str(Path("<agent_work>") / path.resolve().relative_to(work_root.resolve()))
    except ValueError:
        return str(path)


def _display_task(task: Path) -> str:
    return _relativize(task)


def _relativize(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root()))
    except ValueError:
        return str(path)

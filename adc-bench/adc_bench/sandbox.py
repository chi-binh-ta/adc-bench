from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .schemas import TestRunResult


RUNNER = r'''
import importlib.util
import json
import pathlib
import sys
import traceback

suite_path = pathlib.Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("adc_suite", suite_path)
module = importlib.util.module_from_spec(spec)
errors = []
passed = 0
total = 0

try:
    assert spec.loader is not None
    spec.loader.exec_module(module)
    tests = [
        (name, getattr(module, name))
        for name in dir(module)
        if name.startswith("test_") and callable(getattr(module, name))
    ]
    for name, fn in sorted(tests):
        total += 1
        try:
            fn()
            passed += 1
        except Exception:
            errors.append(name + ":\n" + traceback.format_exc(limit=8))
except Exception:
    total = max(total, 1)
    errors.append("suite import failed:\n" + traceback.format_exc(limit=8))

result = {
    "passed": passed == total and total > 0,
    "passed_count": passed,
    "total_count": total,
    "errors": errors,
}
print(json.dumps(result))
sys.exit(0 if result["passed"] else 1)
'''


def run_test_suite(
    *,
    suite_name: str,
    task_dir: Path,
    solution_path: Path,
    timeout_seconds: float = 3.0,
) -> TestRunResult:
    suite_path = task_dir / suite_name
    if not suite_path.exists():
        return TestRunResult(
            suite=suite_name,
            passed=False,
            passed_count=0,
            total_count=0,
            timed_out=False,
            seconds=0.0,
            errors=[f"missing test suite {suite_path}"],
        )

    with tempfile.TemporaryDirectory(prefix="adc_bench_") as temp_name:
        temp_dir = Path(temp_name)
        shutil.copy2(solution_path, temp_dir / "solution.py")
        copied_suite = temp_dir / suite_name
        shutil.copy2(suite_path, copied_suite)
        runner_path = temp_dir / "_adc_runner.py"
        runner_path.write_text(RUNNER, encoding="utf-8")

        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path), str(copied_suite)],
                cwd=temp_dir,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            seconds = time.perf_counter() - start
        except subprocess.TimeoutExpired as exc:
            return TestRunResult(
                suite=suite_name,
                passed=False,
                passed_count=0,
                total_count=1,
                timed_out=True,
                seconds=timeout_seconds,
                errors=[f"{suite_name} timed out after {timeout_seconds:.1f}s"],
                stdout=_sanitize_text(exc.stdout or "", temp_dir),
                stderr=_sanitize_text(exc.stderr or "", temp_dir),
            )

        stdout = _sanitize_text(completed.stdout, temp_dir)
        stderr = _sanitize_text(completed.stderr, temp_dir)

    payload = _parse_runner_json(stdout)
    if payload is None:
        return TestRunResult(
            suite=suite_name,
            passed=False,
            passed_count=0,
            total_count=1,
            timed_out=False,
            seconds=seconds,
            errors=["test runner did not emit valid JSON"],
            stdout=stdout,
            stderr=stderr,
        )

    errors = [_sanitize_text(str(error), temp_dir) for error in payload.get("errors", [])]
    if stderr.strip():
        errors.append(stderr.strip())
    return TestRunResult(
        suite=suite_name,
        passed=bool(payload.get("passed", False)),
        passed_count=int(payload.get("passed_count", 0)),
        total_count=int(payload.get("total_count", 0)),
        timed_out=False,
        seconds=seconds,
        errors=errors,
        stdout=stdout,
        stderr=stderr,
    )


def _parse_runner_json(stdout: str) -> dict[str, object] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _sanitize_text(text: str, temp_dir: Path) -> str:
    if not text:
        return ""
    variants = {str(temp_dir), str(temp_dir).replace("\\", "\\\\")}
    sanitized = text
    for variant in variants:
        sanitized = sanitized.replace(variant, "<sandbox>")
    return sanitized

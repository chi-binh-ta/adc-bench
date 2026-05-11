from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUSPICIOUS_STRINGS = [
    "hidden_tests",
    "stress_tests",
    "reference_solution",
    "expected_trace",
    "os.listdir",
    "os.walk",
    "glob(",
    "pathlib",
    "__file__",
    "inspect",
    "subprocess",
    "eval(",
    "exec(",
]


@dataclass
class VerificationResult:
    task_id: str
    task_dir: str
    submission_dir: str
    passed_public: bool
    passed_hidden: bool
    passed_stress: bool
    correctness: float
    complexity: float
    trace_quality: float
    anti_cheat: float
    adc_score_partial: float
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return {"__json_error__": str(exc)}


def normalize_text(value: Any) -> str:
    return str(value or "").lower().strip()


def keyword_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_text(x) for x in value if normalize_text(x)]
    if isinstance(value, str):
        return [normalize_text(x) for x in value.replace(",", " ").split() if normalize_text(x)]
    return []


def words_from_text(text: str) -> list[str]:
    cleaned = (
        text.lower()
        .replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("+", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
    )
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "with", "by", "is", "are", "be", "as", "at", "from", "into",
        "algorithm", "method", "approach", "structure"
    }
    return [w for w in cleaned.split() if len(w) >= 3 and w not in stopwords]


def complexity_signals(text: str) -> set[str]:
    """Return coarse complexity classes mentioned in a free-form string."""
    t = normalize_text(text)
    compact = (
        t.replace(" ", "")
        .replace("\\", "")
        .replace("*", "")
        .replace("time", "")
        .replace("complexity", "")
        .replace(":", "")
        .replace("=", "")
    )

    signals: set[str] = set()

    if not t:
        return signals

    if "constant" in t or "o(1)" in compact:
        signals.add("constant")

    if "linear" in t or "o(n)" in compact or "theta(n)" in compact:
        signals.add("linear")

    if (
        "graph linear" in t
        or "o(v+e)" in compact
        or "o(n+m)" in compact
        or "vertices+edges" in compact
        or "nodes+edges" in compact
    ):
        signals.add("graph_linear")

    if "logarithmic" in t or "o(logn)" in compact:
        signals.add("logarithmic")

    if (
        "n log n" in t
        or "nlogn" in compact
        or "o(nlogn)" in compact
        or "o(nlog(n))" in compact
    ):
        signals.add("n_log_n")

    if "quadratic" in t or "o(n^2)" in compact or "o(n2)" in compact:
        signals.add("quadratic")

    if "cubic" in t or "o(n^3)" in compact or "o(n3)" in compact:
        signals.add("cubic")

    if "exponential" in t or "2^n" in compact or "o(2^n)" in compact:
        signals.add("exponential")

    return signals


def complexity_matches(actual: str, expected: str) -> bool:
    """Heuristic equivalence check for complexity strings.

    Examples considered equivalent:
    - "O(n)", "O(n) time", "linear time"
    - "O(V + E)", "graph linear time", "vertices + edges"
    - "O(n log n)", "n log n time"
    """
    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)

    if not actual_norm or not expected_norm:
        return False

    actual_compact = actual_norm.replace(" ", "").replace("\\", "")
    expected_compact = expected_norm.replace(" ", "").replace("\\", "")

    if expected_compact and expected_compact in actual_compact:
        return True

    actual_signals = complexity_signals(actual_norm)
    expected_signals = complexity_signals(expected_norm)

    return bool(actual_signals & expected_signals)


def scan_for_cheating(solution_path: Path) -> tuple[float, list[str]]:
    if not solution_path.exists():
        return 0.0, ["solution.py does not exist"]

    text = solution_path.read_text(encoding="utf-8", errors="ignore")
    violations: list[str] = []

    for token in SUSPICIOUS_STRINGS:
        if token in text:
            violations.append(f"Suspicious token found in solution.py: {token}")

    if violations:
        return 0.0, violations

    return 1.0, []


def run_test_file(
    test_file: Path,
    workdir: Path,
    timeout_seconds: int = 5,
) -> tuple[bool, str]:
    if not test_file.exists():
        return False, f"Missing test file: {test_file}"

    local_test = workdir / test_file.name
    shutil.copy2(test_file, local_test)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(local_test.name)],
            cwd=workdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timeout while running {test_file.name}"

    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += "\n" + proc.stderr

    if proc.returncode != 0:
        return False, output.strip()

    return True, output.strip()


def infer_expected_keywords(
    expected_trace: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[list[str], list[str], str]:
    target_keywords = keyword_list(expected_trace.get("target_keywords"))
    invariant_keywords = keyword_list(expected_trace.get("invariant_keywords"))

    if not target_keywords:
        target_algorithm = normalize_text(
            expected_trace.get("chosen_algorithm")
            or expected_trace.get("target_algorithm")
            or metadata.get("target_algorithm")
        )
        target_keywords = words_from_text(target_algorithm)

    if not invariant_keywords:
        invariant = normalize_text(expected_trace.get("invariant"))
        invariant_keywords = words_from_text(invariant)

    expected_complexity = normalize_text(
        expected_trace.get("expected_complexity")
        or expected_trace.get("complexity_time")
        or metadata.get("expected_complexity")
    )

    return target_keywords, invariant_keywords, expected_complexity


def rough_trace_score(
    trace_path: Path,
    expected_trace_path: Path,
    metadata_path: Path,
) -> tuple[float, list[str]]:
    errors: list[str] = []

    if not trace_path.exists():
        return 0.0, ["trace.json does not exist"]

    trace = load_json(trace_path)
    expected_trace = load_json(expected_trace_path)
    metadata = load_json(metadata_path)

    if "__json_error__" in trace:
        return 0.0, [f"trace.json is invalid JSON: {trace['__json_error__']}"]

    if "__json_error__" in expected_trace:
        errors.append(f"expected_trace.json is invalid JSON: {expected_trace['__json_error__']}")

    target_keywords, invariant_keywords, expected_complexity = infer_expected_keywords(
        expected_trace,
        metadata,
    )

    score = 0.0

    chosen_algorithm = normalize_text(trace.get("chosen_algorithm"))
    if chosen_algorithm and target_keywords and any(k in chosen_algorithm for k in target_keywords):
        score += 0.35
    elif chosen_algorithm:
        score += 0.15
        errors.append("chosen_algorithm is present but does not strongly match expected keywords")
    else:
        errors.append("chosen_algorithm is missing")

    invariant = normalize_text(trace.get("invariant"))
    if invariant and invariant_keywords and any(k in invariant for k in invariant_keywords):
        score += 0.25
    elif invariant:
        score += 0.10
        errors.append("invariant is present but weak or does not match expected invariant keywords")
    else:
        errors.append("invariant is missing")

    complexity_time = normalize_text(trace.get("complexity_time"))
    if complexity_matches(complexity_time, expected_complexity):
        score += 0.15
    elif complexity_time:
        score += 0.05
        errors.append("complexity_time is present but does not clearly match expected complexity")
    else:
        errors.append("complexity_time is missing")

    rejected = trace.get("rejected_algorithms")
    if isinstance(rejected, list) and len(rejected) > 0:
        score += 0.15
    else:
        errors.append("rejected_algorithms is missing or empty")

    edge_cases = trace.get("edge_cases")
    if isinstance(edge_cases, list) and len(edge_cases) > 0:
        score += 0.10
    else:
        errors.append("edge_cases is missing or empty")

    return min(score, 1.0), errors


def prepare_workdir(task_dir: Path, submission_dir: Path, workdir: Path) -> list[str]:
    errors: list[str] = []

    solution_path = submission_dir / "solution.py"
    starter_path = task_dir / "starter.py"

    if solution_path.exists():
        shutil.copy2(solution_path, workdir / "solution.py")
    elif starter_path.exists():
        shutil.copy2(starter_path, workdir / "solution.py")
        errors.append("submission solution.py missing; used starter.py as fallback")
    else:
        errors.append("Neither submission solution.py nor starter.py exists")

    return errors


def verify_task(
    task_dir: Path,
    submission_dir: Path,
    timeout_seconds: int = 5,
) -> VerificationResult:
    task_dir = task_dir.resolve()
    submission_dir = submission_dir.resolve()

    metadata = load_json(task_dir / "metadata.json")
    task_id = str(metadata.get("task_id") or task_dir.name)

    errors: list[str] = []

    solution_path = submission_dir / "solution.py"
    trace_path = submission_dir / "trace.json"

    if not submission_dir.exists():
        errors.append(f"Submission directory does not exist: {submission_dir}")

    if not solution_path.exists():
        errors.append("Missing submission solution.py")

    if not trace_path.exists():
        errors.append("Missing submission trace.json")

    anti_cheat, cheat_errors = scan_for_cheating(solution_path)
    errors.extend(cheat_errors)

    passed_public = False
    passed_hidden = False
    passed_stress = False

    with tempfile.TemporaryDirectory(prefix="adc_verify_") as tmp:
        workdir = Path(tmp)
        errors.extend(prepare_workdir(task_dir, submission_dir, workdir))

        passed_public, public_msg = run_test_file(
            task_dir / "public_tests.py",
            workdir,
            timeout_seconds=timeout_seconds,
        )
        if not passed_public:
            errors.append(f"public_tests failed: {public_msg}")

        passed_hidden, hidden_msg = run_test_file(
            task_dir / "hidden_tests.py",
            workdir,
            timeout_seconds=timeout_seconds,
        )
        if not passed_hidden:
            errors.append(f"hidden_tests failed: {hidden_msg}")

        passed_stress, stress_msg = run_test_file(
            task_dir / "stress_tests.py",
            workdir,
            timeout_seconds=timeout_seconds,
        )
        if not passed_stress:
            errors.append(f"stress_tests failed: {stress_msg}")

    trace_quality, trace_errors = rough_trace_score(
        trace_path,
        task_dir / "expected_trace.json",
        task_dir / "metadata.json",
    )
    errors.extend(trace_errors)

    correctness = 1.0 if passed_hidden else 0.0
    complexity = 1.0 if passed_stress else 0.0

    # Partial ADC score for a single task.
    # Transfer and robustness are better computed at evaluator/family level.
    adc_score_partial = (
        0.35 * correctness
        + 0.20 * complexity
        + 0.15 * trace_quality
        + 0.05 * anti_cheat
    )

    return VerificationResult(
        task_id=task_id,
        task_dir=str(task_dir),
        submission_dir=str(submission_dir),
        passed_public=passed_public,
        passed_hidden=passed_hidden,
        passed_stress=passed_stress,
        correctness=correctness,
        complexity=complexity,
        trace_quality=trace_quality,
        anti_cheat=anti_cheat,
        adc_score_partial=round(adc_score_partial, 4),
        errors=errors,
    )



def format_text_report(result: VerificationResult) -> str:
    """Return a readable text report for one verification result."""
    def mark(value: bool) -> str:
        return "PASS" if value else "FAIL"

    lines: list[str] = []
    lines.append("ADC-bench Verification Report")
    lines.append("=" * 34)
    lines.append(f"Task: {result.task_id}")
    lines.append(f"Task dir: {result.task_dir}")
    lines.append(f"Submission dir: {result.submission_dir}")
    lines.append("")
    lines.append("Test results")
    lines.append("-" * 12)
    lines.append(f"public tests : {mark(result.passed_public)}")
    lines.append(f"hidden tests : {mark(result.passed_hidden)}")
    lines.append(f"stress tests : {mark(result.passed_stress)}")
    lines.append("")
    lines.append("Score components")
    lines.append("-" * 16)
    lines.append(f"correctness       : {result.correctness:.3f}")
    lines.append(f"complexity        : {result.complexity:.3f}")
    lines.append(f"trace_quality     : {result.trace_quality:.3f}")
    lines.append(f"anti_cheat        : {result.anti_cheat:.3f}")
    lines.append(f"adc_score_partial : {result.adc_score_partial:.3f}")
    lines.append("")
    lines.append("Errors / warnings")
    lines.append("-" * 17)

    if result.errors:
        for item in result.errors:
            lines.append(f"- {item}")
    else:
        lines.append("None")

    return "\n".join(lines)

def main() -> None:
    if len(sys.argv) not in {3, 4}:
        print(
            "Usage: python -m adc_bench.verifier <task_dir> <submission_dir> [timeout_seconds]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    task_dir = Path(sys.argv[1])
    submission_dir = Path(sys.argv[2])
    timeout_seconds = int(sys.argv[3]) if len(sys.argv) == 4 else 5

    result = verify_task(
        task_dir=task_dir,
        submission_dir=submission_dir,
        timeout_seconds=timeout_seconds,
    )

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

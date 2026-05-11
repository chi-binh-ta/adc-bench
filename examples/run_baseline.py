from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adc_bench.evaluator import discover_tasks, run_task  # noqa: E402
from adc_bench.schemas import TaskMetadata  # noqa: E402
from agents.agent_v0.agent import generate_submission  # noqa: E402


SELECTED_TASKS = {
    "known/two_sum_hash",
    "known/valid_parentheses_stack",
    "synthetic/rotating_key_reachability",
    "adversarial/wrong_hint_shortest_path",
}


def main() -> None:
    print("Running agent_v0 weak baseline")
    for task_dir in discover_tasks(ROOT / "tasks"):
        metadata = TaskMetadata.from_file(task_dir / "metadata.json")
        if metadata.task_id not in SELECTED_TASKS:
            continue
        submission_dir = ROOT / "submissions" / "agent_v0" / task_dir.name
        generate_submission(task_dir, submission_dir)
        result = run_task(task_dir, submission_dir)
        print(
            f"{metadata.task_id:42} "
            f"ADCScore={result['adc_score']:.3f} "
            f"hidden={result['correctness']:.2f} "
            f"stress={'pass' if result['passed_stress'] else 'fail'} "
            f"trace={result['trace_quality']:.2f}"
        )


if __name__ == "__main__":
    main()

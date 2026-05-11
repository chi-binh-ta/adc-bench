from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .evaluator import discover_tasks, project_root, run_all, run_task
from .schemas import TaskMetadata


def _display_path(path: Path) -> str:
    try:
        text = str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        text = str(path)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def main() -> None:
    parser = argparse.ArgumentParser(prog="adc-bench", description="ADC-bench v0.2 evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List benchmark tasks")
    list_parser.add_argument("--tasks", type=Path, default=project_root() / "tasks")

    run_task_parser = subparsers.add_parser("run-task", help="Run one task against a submission")
    run_task_parser.add_argument("--task", type=Path, required=True)
    run_task_parser.add_argument("--submission", type=Path, required=True)
    run_task_parser.add_argument("--timeout", type=float, default=3.0)

    run_all_parser = subparsers.add_parser("run-all", help="Run all tasks and write reports")
    run_all_parser.add_argument("--submissions", type=Path, required=True)
    run_all_parser.add_argument("--tasks", type=Path, default=project_root() / "tasks")
    run_all_parser.add_argument("--reports", type=Path, default=project_root() / "reports")
    run_all_parser.add_argument("--timeout", type=float, default=3.0)

    args = parser.parse_args()
    if args.command == "list":
        for task in discover_tasks(args.tasks):
            metadata = TaskMetadata.from_file(task / "metadata.json")
            print(
                f"{metadata.task_id:42} "
                f"split={metadata.split:11} "
                f"difficulty={metadata.difficulty:6} "
                f"target={metadata.target_algorithm}"
            )
    elif args.command == "run-task":
        result = run_task(args.task, args.submission, timeout_seconds=args.timeout)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "run-all":
        results = run_all(
            submissions_root=args.submissions,
            tasks_root=args.tasks,
            reports_root=args.reports,
            timeout_seconds=args.timeout,
        )
        average = sum(item["adc_score"] for item in results) / len(results) if results else 0.0
        print(f"Ran {len(results)} tasks. Average ADCScore: {average:.3f}")
        print(f"Wrote {_display_path(args.reports / 'results.json')}")
        print(f"Wrote {_display_path(args.reports / 'summary.md')}")


if __name__ == "__main__":
    main()

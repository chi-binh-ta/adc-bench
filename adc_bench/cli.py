from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluator import discover_tasks, project_root, run_all, run_task
from .schemas import TaskMetadata
from .self_improve import evaluate_self_improvement, format_self_improve_text_report
from .verifier import format_text_report, verify_task


def _display_path(path: Path) -> str:
    try:
        text = str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        text = str(path)

    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _write_or_print(content: str, output_path: Path | None) -> None:
    if output_path is None:
        encoding = sys.stdout.encoding or "utf-8"
        safe_content = content.encode(encoding, errors="replace").decode(encoding)
        print(safe_content)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote {_display_path(output_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="adc-bench",
        description="ADC-bench evaluator and verifier",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List benchmark tasks")
    list_parser.add_argument("--tasks", type=Path, default=project_root() / "tasks")

    run_task_parser = subparsers.add_parser(
        "run-task",
        help="Run one task against a submission using the benchmark evaluator",
    )
    run_task_parser.add_argument("--task", type=Path, required=True)
    run_task_parser.add_argument("--submission", type=Path, required=True)
    run_task_parser.add_argument("--timeout", type=float, default=3.0)

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Run all tasks and write reports",
    )
    run_all_parser.add_argument("--submissions", type=Path, required=True)
    run_all_parser.add_argument("--tasks", type=Path, default=project_root() / "tasks")
    run_all_parser.add_argument("--reports", type=Path, default=project_root() / "reports")
    run_all_parser.add_argument("--timeout", type=float, default=3.0)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify one task submission with public, hidden, stress, trace, and anti-cheat checks",
    )
    verify_parser.add_argument("--task", type=Path, required=True)
    verify_parser.add_argument("--submission", type=Path, required=True)
    verify_parser.add_argument("--timeout", type=float, default=5.0)
    verify_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format for the verifier report",
    )
    verify_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the verifier report",
    )

    self_improve_parser = subparsers.add_parser(
        "self-improve-eval",
        help="Evaluate an offline candidate agent improvement against a baseline agent",
    )
    self_improve_parser.add_argument("--baseline-agent", type=Path, required=True)
    self_improve_parser.add_argument("--candidate-agent", type=Path, required=True)
    self_improve_parser.add_argument("--task", type=Path, required=True)
    self_improve_parser.add_argument("--timeout", type=float, default=5.0)
    self_improve_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format for the self-improvement report",
    )
    self_improve_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the self-improvement report",
    )

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
        result = run_task(
            args.task,
            args.submission,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(result, indent=2, sort_keys=True))

    elif args.command == "run-all":
        results = run_all(
            submissions_root=args.submissions,
            tasks_root=args.tasks,
            reports_root=args.reports,
            timeout_seconds=args.timeout,
        )

        average = (
            sum(item["adc_score"] for item in results) / len(results)
            if results
            else 0.0
        )

        print(f"Ran {len(results)} tasks.")
        print(f"Average ADCScore: {average:.3f}")
        print(f"Wrote {_display_path(args.reports / 'results.json')}")
        print(f"Wrote {_display_path(args.reports / 'summary.md')}")

    elif args.command == "verify":
        result = verify_task(
            task_dir=args.task,
            submission_dir=args.submission,
            timeout_seconds=args.timeout,
        )

        if args.format == "text":
            content = format_text_report(result)
        else:
            content = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

        _write_or_print(content, args.out)

    elif args.command == "self-improve-eval":
        result = evaluate_self_improvement(
            baseline_agent_dir=args.baseline_agent,
            candidate_agent_dir=args.candidate_agent,
            self_improve_task_dir=args.task,
            timeout_seconds=args.timeout,
        )

        if args.format == "text":
            content = format_self_improve_text_report(result)
        else:
            content = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

        _write_or_print(content, args.out)


if __name__ == "__main__":
    main()

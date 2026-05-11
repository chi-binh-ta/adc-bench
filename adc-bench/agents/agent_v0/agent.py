from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


WEAK_TRACE = {
    "chosen_algorithm": "direct simulation",
    "hypotheses": [
        "The starter implementation may work for small visible examples.",
        "A better algorithm may be needed for hidden or stress cases."
    ],
    "rejected_algorithms": [],
    "invariant": "No strong invariant discovered.",
    "complexity_time": "unknown",
    "complexity_space": "unknown",
    "edge_cases": ["empty input", "small input"],
    "counterexample_for_wrong_approach": ""
}


def generate_submission(task_dir: Path, submission_dir: Path) -> None:
    task_dir = task_dir.resolve()
    submission_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task_dir / "starter.py", submission_dir / "solution.py")
    with (submission_dir / "trace.json").open("w", encoding="utf-8") as handle:
        json.dump(WEAK_TRACE, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weak ADC-bench baseline submission")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    generate_submission(args.task, args.submission)
    print(f"Wrote weak baseline submission to {args.submission}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


VALID_SPLITS = {"known", "composed", "synthetic", "adversarial"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass(frozen=True)
class TaskMetadata:
    task_id: str
    family: str
    split: str
    target_algorithm: str
    expected_complexity: str
    difficulty: str
    requires_invariant: bool
    requires_counterexample: bool
    anti_cheat: bool

    @classmethod
    def from_file(cls, path: Path) -> "TaskMetadata":
        data = load_json(path)
        missing = {
            "task_id",
            "family",
            "split",
            "target_algorithm",
            "expected_complexity",
            "difficulty",
            "requires_invariant",
            "requires_counterexample",
            "anti_cheat",
        } - set(data)
        if missing:
            raise ValueError(f"{path} missing metadata fields: {sorted(missing)}")
        metadata = cls(**{field: data[field] for field in cls.__dataclass_fields__})
        metadata.validate(path)
        return metadata

    def validate(self, path: Path | None = None) -> None:
        label = str(path) if path else self.task_id
        if self.split not in VALID_SPLITS:
            raise ValueError(f"{label}: invalid split {self.split!r}")
        if self.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"{label}: invalid difficulty {self.difficulty!r}")
        if not self.task_id or "/" not in self.task_id:
            raise ValueError(f"{label}: task_id should look like '<split>/<name>'")


@dataclass(frozen=True)
class TraceValidationResult:
    score: float
    details: dict[str, float]
    errors: list[str]


@dataclass(frozen=True)
class TestRunResult:
    suite: str
    passed: bool
    passed_count: int
    total_count: int
    timed_out: bool
    seconds: float
    errors: list[str]
    stdout: str = ""
    stderr: str = ""

    @property
    def pass_ratio(self) -> float:
        if self.total_count <= 0:
            return 0.0
        return self.passed_count / self.total_count


@dataclass(frozen=True)
class AntiCheatResult:
    passed: bool
    score: float
    violations: list[str]


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")

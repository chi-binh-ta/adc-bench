from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import write_json


def write_reports(results: list[dict[str, Any]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)
    write_json(
        report_dir / "results.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "tasks": results,
        },
    )
    (report_dir / "summary.md").write_text(render_summary_markdown(results, summary), encoding="utf-8")


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["adc_score", "correctness", "complexity", "trace_quality"]
    summary: dict[str, Any] = {"total_tasks": len(results)}
    for field in fields:
        summary[f"average_{field}"] = _average(results, field)
    summary["anti_cheat_violations"] = [
        item["task_id"] for item in results if float(item.get("anti_cheat", 0.0)) < 1.0
    ]
    summary["top_failed_tasks"] = [
        item["task_id"]
        for item in sorted(results, key=lambda result: (result.get("adc_score", 0.0), result["task_id"]))[:5]
    ]
    return summary


def render_summary_markdown(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# ADC-bench Summary",
        "",
        f"- Total tasks: {summary['total_tasks']}",
        f"- Average ADCScore: {summary['average_adc_score']:.3f}",
        f"- Average correctness: {summary['average_correctness']:.3f}",
        f"- Average complexity: {summary['average_complexity']:.3f}",
        f"- Average trace quality: {summary['average_trace_quality']:.3f}",
        "",
        "## Top Failed Tasks",
        "",
    ]
    failed = summary.get("top_failed_tasks", [])
    if failed:
        by_task = {item["task_id"]: item for item in results}
        for task_id in failed:
            lines.append(f"- {task_id}: ADCScore {by_task[task_id]['adc_score']:.3f}")
    else:
        lines.append("- None")

    lines.extend(["", "## Anti-cheat Violations", ""])
    violations = summary.get("anti_cheat_violations", [])
    if violations:
        for task_id in violations:
            lines.append(f"- {task_id}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _average(results: list[dict[str, Any]], field: str) -> float:
    if not results:
        return 0.0
    return round(sum(float(item.get(field, 0.0)) for item in results) / len(results), 6)

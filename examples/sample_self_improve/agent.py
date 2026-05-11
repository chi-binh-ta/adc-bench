from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def solve_two_sum_source() -> str:
    return '''from __future__ import annotations


def solve(nums: list[int], target: int) -> bool:
    seen: set[int] = set()
    for value in nums:
        if target - value in seen:
            return True
        seen.add(value)
    return False
'''


def solve_parentheses_source() -> str:
    return '''from __future__ import annotations


def solve(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        else:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack
'''


def solve_graph_reachability_source() -> str:
    return '''from __future__ import annotations

from collections import deque


def solve(n: int, edges: list[tuple[int, int]], source: int, target: int) -> bool:
    graph: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    seen = [False] * n
    seen[source] = True
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for nxt in graph[node]:
            if not seen[nxt]:
                seen[nxt] = True
                queue.append(nxt)
    return False
'''


def solve_dijkstra_source() -> str:
    return '''from __future__ import annotations

from heapq import heappop, heappush


def solve(n: int, edges: list[tuple[int, int, int]], source: int, target: int) -> int:
    graph: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, weight in edges:
        graph[u].append((v, weight))

    dist = [float("inf")] * n
    dist[source] = 0
    heap: list[tuple[int, int]] = [(0, source)]
    while heap:
        cost, node = heappop(heap)
        if node == target:
            return cost
        if cost != dist[node]:
            continue
        for nxt, weight in graph[node]:
            new_cost = cost + weight
            if new_cost < dist[nxt]:
                dist[nxt] = new_cost
                heappush(heap, (new_cost, nxt))
    return -1
'''


TRACE_BY_TASK: dict[str, dict[str, Any]] = {
    "known/two_sum_hash": {
        "chosen_algorithm": "hash set complement lookup",
        "hypotheses": [
            "try nested loops",
            "sort and use two pointers",
            "scan once while storing seen values in a hash set"
        ],
        "rejected_algorithms": [
            {
                "name": "brute force nested loops",
                "reason": "Quadratic pair scans do not meet the linear complexity target."
            }
        ],
        "invariant": "After each processed value, seen contains exactly earlier values; a complement in seen proves two distinct positions sum to the target.",
        "complexity_time": "O(n)",
        "complexity_space": "O(n)",
        "edge_cases": ["duplicate values", "negative numbers", "single element", "no pair"],
        "counterexample_for_wrong_approach": "Using one element twice would accept nums=[5], target=10."
    },
    "known/valid_parentheses_stack": {
        "chosen_algorithm": "stack-based nested bracket matching",
        "hypotheses": [
            "count opening and closing brackets",
            "track the last unmatched opening bracket with a stack"
        ],
        "rejected_algorithms": [
            {
                "name": "count-based matching",
                "reason": "Counts ignore order, so cross-nested strings can be accepted incorrectly."
            }
        ],
        "invariant": "The stack contains unmatched opening brackets in nesting order; each closing bracket must match the last opening bracket.",
        "complexity_time": "O(n)",
        "complexity_space": "O(n)",
        "edge_cases": ["empty string", "closing before opening", "([)]", "deep nesting"],
        "counterexample_for_wrong_approach": "([)] has balanced counts but invalid nesting order."
    },
    "known/graph_reachability_bfs": {
        "chosen_algorithm": "BFS frontier graph reachability",
        "hypotheses": [
            "check only the direct edge",
            "explore reachable nodes with BFS or DFS"
        ],
        "rejected_algorithms": [
            {
                "name": "direct edge check",
                "reason": "Reachability can require multiple directed hops."
            }
        ],
        "invariant": "Every visited node is reachable from the source, and the queue stores the current frontier of reachable nodes whose outgoing edges remain to be explored.",
        "complexity_time": "O(n + m)",
        "complexity_space": "O(n + m)",
        "edge_cases": ["source equals target", "cycle", "disconnected component", "long chain"],
        "counterexample_for_wrong_approach": "A path 0->1->2 reaches 2 without a direct edge 0->2."
    },
    "known/dijkstra_positive_weights": {
        "chosen_algorithm": "Dijkstra priority heap relaxation",
        "hypotheses": [
            "treat the graph like an unweighted BFS",
            "relax positive weighted edges with a priority heap"
        ],
        "rejected_algorithms": [
            {
                "name": "BFS by hop count",
                "reason": "A path with fewer hops can be more expensive when edge weights differ."
            }
        ],
        "invariant": "When a node is popped with its current best cost, that settled cost is the shortest known path; every relaxation only improves a neighbor distance.",
        "complexity_time": "O((n + m) log n)",
        "complexity_space": "O(n + m)",
        "edge_cases": ["unreachable target", "weighted detour beats one hop", "stale heap entries"],
        "counterexample_for_wrong_approach": "Edges 0->1 cost 10 and 0->2->1 costs 2 make hop-count BFS wrong."
    },
}


SOURCE_BY_TASK = {
    "known/two_sum_hash": solve_two_sum_source,
    "known/valid_parentheses_stack": solve_parentheses_source,
    "known/graph_reachability_bfs": solve_graph_reachability_source,
    "known/dijkstra_positive_weights": solve_dijkstra_source,
}


def weak_trace(task_id: str, problem_text: str) -> dict[str, Any]:
    return {
        "chosen_algorithm": "starter fallback",
        "hypotheses": [
            f"No specialized template matched {task_id}.",
            f"Visible problem length was {len(problem_text)} characters."
        ],
        "rejected_algorithms": [],
        "invariant": "No strong invariant inferred by the sample candidate.",
        "complexity_time": "unknown",
        "complexity_space": "unknown",
        "edge_cases": ["small input"],
        "counterexample_for_wrong_approach": ""
    }


def generate_submission(task_dir: Path, out_dir: Path) -> None:
    task_dir = task_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8-sig"))
    problem_text = (task_dir / "problem.md").read_text(encoding="utf-8-sig")
    task_id = str(metadata.get("task_id", task_dir.name))

    source_factory = SOURCE_BY_TASK.get(task_id)
    if source_factory is None:
        shutil.copy2(task_dir / "starter.py", out_dir / "solution.py")
        trace = weak_trace(task_id, problem_text)
    else:
        (out_dir / "solution.py").write_text(source_factory(), encoding="utf-8")
        trace = TRACE_BY_TASK[task_id]

    with (out_dir / "trace.json").open("w", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample self-improved ADC submission")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate_submission(args.task, args.out)
    print(f"Wrote sample candidate submission to {args.out}")


if __name__ == "__main__":
    main()

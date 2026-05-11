# Wrong Hint Shortest Path

Implement `solve(n: int, roads: list[tuple[int, int, int]], start: int, target: int) -> int`.

The graph is directed. Every road `(u, v, cost)` has a positive traversal cost.
Return the minimum total cost from `start` to `target`, or `-1` if unreachable.

Misleading hint: This probably needs BFS.

That hint is intentionally unsafe. The correct structure is Dijkstra's algorithm
because the graph is weighted.

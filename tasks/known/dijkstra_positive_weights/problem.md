# Dijkstra Positive Weights

Implement `solve(n: int, edges: list[tuple[int, int, int]], source: int, target: int) -> int`.

The graph is directed, nodes are `0..n-1`, and every edge weight is positive.
Return the minimum total weight from `source` to `target`, or `-1` if no path
exists.

The intended algorithm is Dijkstra's priority-queue relaxation. A BFS-style
minimum-hop search is incorrect when weights differ.

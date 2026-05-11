# Graph Reachability BFS

Implement `solve(n: int, edges: list[tuple[int, int]], source: int, target: int) -> bool`.

The graph is directed and contains nodes `0..n-1`. Return whether `target` is
reachable from `source` by following zero or more directed edges.

The intended structure is breadth-first search or depth-first search over the
reachable frontier, not merely checking for a direct edge.

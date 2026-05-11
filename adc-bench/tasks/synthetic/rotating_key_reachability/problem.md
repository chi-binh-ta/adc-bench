# Rotating Key Reachability

Implement
`solve(n: int, edges: list[tuple[int, int, int]], start: int, target: int, k: int) -> int`.

The graph is directed. Each edge `(u, v, required_mod)` may be traversed only at
times `t` where `t % k == required_mod`. Traversing an edge takes one time unit.
At any node, you may also wait for one time unit. You start at `start` at time
`0`.

Return the minimum arrival time at `target`, or `-1` if no state can reach it.

The intended structure is BFS on the expanded state graph `(node, time_mod)`.
Normal graph reachability ignores the rotating key state and is insufficient.

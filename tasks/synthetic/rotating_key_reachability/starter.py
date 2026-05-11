from collections import deque


def solve(n: int, edges: list[tuple[int, int, int]], start: int, target: int, k: int) -> int:
    graph: list[list[int]] = [[] for _ in range(n)]
    for u, v, _required_mod in edges:
        graph[u].append(v)

    dist = [-1] * n
    dist[start] = 0
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        if node == target:
            return dist[node]
        for nxt in graph[node]:
            if dist[nxt] == -1:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return -1

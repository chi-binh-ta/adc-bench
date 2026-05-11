from collections import deque


def solve(n: int, edges: list[tuple[int, int, int]], source: int, target: int) -> int:
    graph: list[list[int]] = [[] for _ in range(n)]
    for u, v, _weight in edges:
        graph[u].append(v)

    dist = [-1] * n
    dist[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if dist[nxt] == -1:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist[target]

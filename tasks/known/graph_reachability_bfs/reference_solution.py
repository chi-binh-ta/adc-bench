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

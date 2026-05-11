from collections import deque


def solve(n: int, edges: list[tuple[int, int, int]], start: int, target: int, k: int) -> int:
    graph: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, required_mod in edges:
        graph[u].append((v, required_mod))

    dist = [[-1] * k for _ in range(n)]
    dist[start][0] = 0
    queue: deque[tuple[int, int]] = deque([(start, 0)])
    while queue:
        node, mod = queue.popleft()
        time = dist[node][mod]
        if node == target:
            return time

        next_mod = (mod + 1) % k
        if dist[node][next_mod] == -1:
            dist[node][next_mod] = time + 1
            queue.append((node, next_mod))

        for nxt, required_mod in graph[node]:
            if mod == required_mod and dist[nxt][next_mod] == -1:
                dist[nxt][next_mod] = time + 1
                queue.append((nxt, next_mod))
    return -1

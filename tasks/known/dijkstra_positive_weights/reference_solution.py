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

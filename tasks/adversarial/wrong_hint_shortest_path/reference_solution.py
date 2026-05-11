from heapq import heappop, heappush


def solve(n: int, roads: list[tuple[int, int, int]], start: int, target: int) -> int:
    graph: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, cost in roads:
        graph[u].append((v, cost))

    dist = [float("inf")] * n
    dist[start] = 0
    heap: list[tuple[int, int]] = [(0, start)]
    while heap:
        cost, node = heappop(heap)
        if node == target:
            return cost
        if cost != dist[node]:
            continue
        for nxt, edge_cost in graph[node]:
            new_cost = cost + edge_cost
            if new_cost < dist[nxt]:
                dist[nxt] = new_cost
                heappush(heap, (new_cost, nxt))
    return -1

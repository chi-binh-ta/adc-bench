def solve(n: int, clauses: list[tuple[tuple[int, bool], tuple[int, bool]]]) -> bool:
    graph: list[list[int]] = [[] for _ in range(2 * n)]
    reverse: list[list[int]] = [[] for _ in range(2 * n)]

    def idx(literal: tuple[int, bool]) -> int:
        var, is_true = literal
        return 2 * var + (1 if is_true else 0)

    def add_edge(a: int, b: int) -> None:
        graph[a].append(b)
        reverse[b].append(a)

    for a, b in clauses:
        ai = idx(a)
        bi = idx(b)
        add_edge(ai ^ 1, bi)
        add_edge(bi ^ 1, ai)

    seen = [False] * (2 * n)
    order: list[int] = []

    def dfs(node: int) -> None:
        seen[node] = True
        for nxt in graph[node]:
            if not seen[nxt]:
                dfs(nxt)
        order.append(node)

    for node in range(2 * n):
        if not seen[node]:
            dfs(node)

    comp = [-1] * (2 * n)

    def assign(node: int, label: int) -> None:
        comp[node] = label
        for nxt in reverse[node]:
            if comp[nxt] == -1:
                assign(nxt, label)

    label = 0
    for node in reversed(order):
        if comp[node] == -1:
            assign(node, label)
            label += 1

    for var in range(n):
        if comp[2 * var] == comp[2 * var + 1]:
            return False
    return True

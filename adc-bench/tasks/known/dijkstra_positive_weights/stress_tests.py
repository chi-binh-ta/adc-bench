from solution import solve


def test_large_weighted_chain_vs_expensive_direct_edges() -> None:
    n = 3000
    edges = [(i, i + 1, 1) for i in range(n - 1)]
    edges.extend((0, i, 100000) for i in range(2, n))
    assert solve(n, edges, 0, n - 1) == n - 1

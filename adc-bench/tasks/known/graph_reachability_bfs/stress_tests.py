from solution import solve


def test_long_chain() -> None:
    n = 30000
    edges = [(i, i + 1) for i in range(n - 1)]
    assert solve(n, edges, 0, n - 1) is True

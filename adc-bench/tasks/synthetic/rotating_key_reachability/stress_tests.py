from solution import solve


def test_long_chain_with_repeated_waits() -> None:
    n = 80
    k = 5
    edges = [(i, i + 1, 0) for i in range(n - 1)]
    assert solve(n, edges, 0, n - 1, k) == 1 + k * (n - 2)

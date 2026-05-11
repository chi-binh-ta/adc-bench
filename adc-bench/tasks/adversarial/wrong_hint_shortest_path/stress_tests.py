from solution import solve


def test_large_adversarial_weight_pattern() -> None:
    n = 2500
    roads = [(i, i + 1, 1) for i in range(n - 1)]
    roads.extend((0, i, 100000) for i in range(2, n))
    assert solve(n, roads, 0, n - 1) == n - 1

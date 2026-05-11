from solution import solve


def test_large_touching_cover_collapses_to_one_interval() -> None:
    n = 50000
    bits = [1 if i % 3 == 0 else 0 for i in range(n)]
    intervals = [(i, min(i + 2, n - 1)) for i in range(0, n, 2)]
    expected = 1 if sum(bits) % 2 else 0
    assert solve(bits, intervals) == expected

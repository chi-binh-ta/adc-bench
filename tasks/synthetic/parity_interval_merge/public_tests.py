from solution import solve


def test_non_overlapping_intervals() -> None:
    bits = [1, 0, 1, 1, 0]
    assert solve(bits, [(0, 1), (3, 4)]) == 2


def test_empty_intervals() -> None:
    assert solve([1, 1, 0], []) == 0


def test_single_interval() -> None:
    assert solve([1, 1, 1], [(0, 2)]) == 1

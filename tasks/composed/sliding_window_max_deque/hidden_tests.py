from solution import solve


def test_duplicate_maxima() -> None:
    assert solve([5, 5, 1, 5, 5], 2) == [5, 5, 5, 5]


def test_negative_values() -> None:
    assert solve([-8, -7, -9, -3], 2) == [-7, -7, -3]


def test_full_window() -> None:
    assert solve([2, 1, 4], 3) == [4]

from solution import solve


def test_classic_example() -> None:
    assert solve([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]


def test_window_size_one() -> None:
    assert solve([4, 2, 9], 1) == [4, 2, 9]


def test_empty_input() -> None:
    assert solve([], 3) == []

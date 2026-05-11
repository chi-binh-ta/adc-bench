from solution import solve


def test_basic_pair_exists() -> None:
    assert solve([2, 7, 11, 15], 9) is True


def test_no_pair() -> None:
    assert solve([1, 2, 4, 8], 20) is False


def test_distinct_positions_required() -> None:
    assert solve([5], 10) is False

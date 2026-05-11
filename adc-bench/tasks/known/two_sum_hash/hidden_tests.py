from solution import solve


def test_duplicates_can_form_pair() -> None:
    assert solve([3, 3], 6) is True


def test_negative_numbers() -> None:
    assert solve([-4, 10, 5, -1], 6) is True


def test_many_near_misses() -> None:
    nums = [10, 20, 30, 40, 50, 60]
    assert solve(nums, 15) is False

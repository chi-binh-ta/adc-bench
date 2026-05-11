from solution import solve


def test_large_no_pair_finishes_quickly() -> None:
    nums = list(range(20000))
    assert solve(nums, -1) is False

from solution import solve


def test_large_sawtooth_input() -> None:
    nums = [i % 997 for i in range(80000)]
    result = solve(nums, 1500)
    assert len(result) == len(nums) - 1500 + 1
    assert result[:10] == [996] * 10

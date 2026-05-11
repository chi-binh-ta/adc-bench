def solve(nums: list[int], target: int) -> bool:
    seen: set[int] = set()
    for value in nums:
        if target - value in seen:
            return True
        seen.add(value)
    return False

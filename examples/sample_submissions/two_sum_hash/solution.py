from __future__ import annotations


def solve(nums: list[int], target: int) -> bool:
    seen: set[int] = set()

    for x in nums:
        complement = target - x
        if complement in seen:
            return True
        seen.add(x)

    return False
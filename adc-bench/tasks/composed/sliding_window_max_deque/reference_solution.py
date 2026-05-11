from collections import deque


def solve(nums: list[int], k: int) -> list[int]:
    if not nums or k <= 0:
        return []
    q: deque[int] = deque()
    out: list[int] = []
    for i, value in enumerate(nums):
        while q and q[0] <= i - k:
            q.popleft()
        while q and nums[q[-1]] <= value:
            q.pop()
        q.append(i)
        if i >= k - 1:
            out.append(nums[q[0]])
    return out

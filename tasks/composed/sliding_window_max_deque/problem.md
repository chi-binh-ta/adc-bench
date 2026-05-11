# Sliding Window Maximum Deque

Implement `solve(nums: list[int], k: int) -> list[int]`.

For each contiguous window of length `k`, return the maximum value in that
window. When `nums` is empty or `k <= 0`, return an empty list.

The intended solution maintains a monotonic deque of candidate indices so each
element is inserted and removed at most once.

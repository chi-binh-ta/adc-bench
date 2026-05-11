def solve(bits: list[int], intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0

    prefix = [0]
    for bit in bits:
        prefix.append(prefix[-1] ^ bit)

    merged: list[tuple[int, int]] = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1] + 1:
            merged.append((left, right))
        else:
            old_left, old_right = merged[-1]
            merged[-1] = (old_left, max(old_right, right))

    odd = 0
    for left, right in merged:
        if prefix[right + 1] ^ prefix[left]:
            odd += 1
    return odd

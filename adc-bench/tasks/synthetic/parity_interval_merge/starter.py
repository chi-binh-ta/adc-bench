def solve(bits: list[int], intervals: list[tuple[int, int]]) -> int:
    count = 0
    for left, right in intervals:
        parity = 0
        for idx in range(left, right + 1):
            parity ^= bits[idx]
        if parity == 1:
            count += 1
    return count

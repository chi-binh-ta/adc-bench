# Parity Interval Merge

Implement `solve(bits: list[int], intervals: list[tuple[int, int]]) -> int`.

`bits` is a list of `0` and `1` values. Each interval is inclusive and valid:
`0 <= left <= right < len(bits)`.

First merge intervals that overlap or touch. For example, `[0, 2]` and `[3, 4]`
become `[0, 4]`. Then return the number of merged intervals whose bitwise XOR
parity is `1`.

The intended solution sorts and merges intervals, then uses prefix parity to
query each merged interval in constant time.

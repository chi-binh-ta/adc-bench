def solve(s: str) -> bool:
    counts = {ch: 0 for ch in "()[]{}"}
    for ch in s:
        counts[ch] += 1
    return counts["("] == counts[")"] and counts["["] == counts["]"] and counts["{"] == counts["}"]

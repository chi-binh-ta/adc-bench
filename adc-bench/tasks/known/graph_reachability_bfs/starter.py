def solve(n: int, edges: list[tuple[int, int]], source: int, target: int) -> bool:
    if source == target:
        return True
    return (source, target) in edges

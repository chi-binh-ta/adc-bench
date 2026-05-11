def solve(n: int, clauses: list[tuple[tuple[int, bool], tuple[int, bool]]]) -> bool:
    def lit_value(literal: tuple[int, bool], mask: int) -> bool:
        var, is_true = literal
        value = bool((mask >> var) & 1)
        return value if is_true else not value

    for mask in range(1 << n):
        if all(lit_value(a, mask) or lit_value(b, mask) for a, b in clauses):
            return True
    return False

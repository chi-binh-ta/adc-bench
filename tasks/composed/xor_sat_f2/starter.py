def solve(n: int, equations: list[tuple[list[int], int]]) -> bool:
    for mask in range(1 << n):
        ok = True
        for variables, rhs in equations:
            value = 0
            for var in variables:
                value ^= (mask >> var) & 1
            if value != rhs:
                ok = False
                break
        if ok:
            return True
    return False

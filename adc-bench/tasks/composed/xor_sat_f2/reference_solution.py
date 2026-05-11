def solve(n: int, equations: list[tuple[list[int], int]]) -> bool:
    rows: list[int] = []
    rhs_bit = 1 << n
    variable_mask = rhs_bit - 1
    for variables, rhs in equations:
        row = 0
        for var in variables:
            row ^= 1 << var
        if rhs:
            row ^= rhs_bit
        rows.append(row)

    rank = 0
    for col in range(n):
        pivot = None
        for r in range(rank, len(rows)):
            if (rows[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and ((rows[r] >> col) & 1):
                rows[r] ^= rows[rank]
        rank += 1

    for row in rows:
        if (row & variable_mask) == 0 and (row & rhs_bit):
            return False
    return True

# 2-SAT SCC

Implement `solve(n: int, clauses: list[tuple[tuple[int, bool], tuple[int, bool]]]) -> bool`.

There are Boolean variables `x0..x(n-1)`. A literal is `(var, is_true)`, where
`(2, False)` means `not x2`. Each clause is a pair of literals representing
`literal_a OR literal_b`.

Return whether all clauses can be satisfied.

The intended algorithm builds the implication graph and checks strongly
connected components. Brute-force assignment enumeration does not scale.

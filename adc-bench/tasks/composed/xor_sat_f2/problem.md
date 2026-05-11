# XOR SAT over F2

Implement `solve(n: int, equations: list[tuple[list[int], int]]) -> bool`.

There are Boolean variables `x0..x(n-1)`. Each equation is `(variables, rhs)`,
meaning the XOR of the listed variables must equal `rhs` (`0` or `1`). A
variable may appear more than once in an equation, in which case repeated
occurrences cancel modulo 2.

Return whether all equations are simultaneously satisfiable.

The intended structure is Gaussian elimination over GF(2), not enumerating all
assignments.

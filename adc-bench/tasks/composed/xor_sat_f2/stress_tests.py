from random import Random

from solution import solve


def test_many_variables_without_bruteforce() -> None:
    rng = Random(12345)
    n = 32
    assignment = [rng.randrange(2) for _ in range(n)]
    equations = []
    for _ in range(90):
        variables = rng.sample(range(n), 5)
        rhs = 0
        for var in variables:
            rhs ^= assignment[var]
        equations.append((variables, rhs))
    assert solve(n, equations) is True

from solution import solve


def test_many_variables_implication_chain() -> None:
    n = 120
    clauses = [((0, True), (0, True))]
    clauses.extend(((i, False), (i + 1, True)) for i in range(n - 1))
    clauses.extend(((i + 1, False), (i, True)) for i in range(n - 1))
    assert solve(n, clauses) is True

from solution import solve


def test_simple_satisfiable() -> None:
    clauses = [((0, True), (1, True)), ((0, False), (1, True))]
    assert solve(2, clauses) is True


def test_single_variable_unsat() -> None:
    clauses = [((0, True), (0, True)), ((0, False), (0, False))]
    assert solve(1, clauses) is False

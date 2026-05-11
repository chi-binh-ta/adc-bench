from solution import solve


def test_implication_cycle_contradiction() -> None:
    clauses = [
        ((0, True), (1, True)),
        ((0, True), (1, False)),
        ((0, False), (1, True)),
        ((0, False), (1, False)),
    ]
    assert solve(2, clauses) is False


def test_larger_satisfiable_formula() -> None:
    clauses = [
        ((0, False), (1, True)),
        ((1, False), (2, True)),
        ((2, False), (3, True)),
        ((3, True), (0, True)),
        ((1, True), (3, False)),
    ]
    assert solve(4, clauses) is True

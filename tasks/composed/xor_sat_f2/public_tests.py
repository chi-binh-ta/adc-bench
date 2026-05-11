from solution import solve


def test_single_equation_satisfiable() -> None:
    assert solve(2, [([0, 1], 1)]) is True


def test_direct_contradiction() -> None:
    assert solve(1, [([0], 0), ([0], 1)]) is False


def test_duplicate_variable_cancels() -> None:
    assert solve(1, [([0, 0], 0)]) is True

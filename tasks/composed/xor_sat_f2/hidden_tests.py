from solution import solve


def test_implied_contradiction() -> None:
    equations = [([0, 1], 0), ([1, 2], 0), ([0, 2], 1)]
    assert solve(3, equations) is False


def test_rank_deficient_consistent_system() -> None:
    equations = [([0, 1], 1), ([1, 2], 0), ([0, 2], 1)]
    assert solve(4, equations) is True

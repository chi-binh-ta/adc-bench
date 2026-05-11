from solution import solve


def test_direct_edge() -> None:
    assert solve(3, [(0, 1), (1, 2)], 0, 1) is True


def test_same_node() -> None:
    assert solve(4, [], 2, 2) is True


def test_not_reachable() -> None:
    assert solve(4, [(0, 1), (2, 3)], 0, 3) is False

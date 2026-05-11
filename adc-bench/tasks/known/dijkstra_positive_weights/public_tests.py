from solution import solve


def test_unit_weights_match_bfs() -> None:
    edges = [(0, 1, 1), (1, 2, 1)]
    assert solve(3, edges, 0, 2) == 2


def test_unreachable() -> None:
    assert solve(4, [(0, 1, 5), (2, 3, 1)], 0, 3) == -1

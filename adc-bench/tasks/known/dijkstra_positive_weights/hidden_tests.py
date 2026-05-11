from solution import solve


def test_weighted_path_beats_one_hop() -> None:
    edges = [(0, 1, 10), (0, 2, 1), (2, 1, 1)]
    assert solve(3, edges, 0, 1) == 2


def test_multiple_relaxations() -> None:
    edges = [(0, 1, 8), (0, 2, 2), (2, 1, 2), (1, 3, 1), (2, 3, 10)]
    assert solve(4, edges, 0, 3) == 5

from solution import solve


def test_reject_bfs_hint_for_weighted_graph() -> None:
    roads = [(0, 1, 100), (0, 2, 1), (2, 1, 1)]
    assert solve(3, roads, 0, 1) == 2


def test_expensive_short_hop_is_wrong() -> None:
    roads = [(0, 3, 50), (0, 1, 4), (1, 2, 4), (2, 3, 4)]
    assert solve(4, roads, 0, 3) == 12

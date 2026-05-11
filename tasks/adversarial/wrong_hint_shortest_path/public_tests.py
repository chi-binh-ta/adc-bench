from solution import solve


def test_unit_weights_make_hint_look_plausible() -> None:
    roads = [(0, 1, 1), (1, 2, 1)]
    assert solve(3, roads, 0, 2) == 2


def test_unreachable() -> None:
    assert solve(3, [(0, 1, 2)], 0, 2) == -1

from solution import solve


def test_no_wait_needed() -> None:
    assert solve(3, [(0, 1, 0), (1, 2, 1)], 0, 2, 3) == 2


def test_unreachable_without_edges() -> None:
    assert solve(2, [], 0, 1, 4) == -1

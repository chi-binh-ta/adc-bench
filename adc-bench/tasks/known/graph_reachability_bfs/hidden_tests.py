from solution import solve


def test_indirect_path() -> None:
    assert solve(5, [(0, 1), (1, 2), (2, 4)], 0, 4) is True


def test_cycle_does_not_loop_forever() -> None:
    assert solve(4, [(0, 1), (1, 2), (2, 1), (2, 3)], 0, 3) is True


def test_disconnected_component() -> None:
    assert solve(6, [(0, 1), (1, 2), (3, 4)], 0, 4) is False

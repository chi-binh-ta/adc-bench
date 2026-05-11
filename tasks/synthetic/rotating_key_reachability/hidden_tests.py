from solution import solve


def test_waiting_for_key_phase() -> None:
    assert solve(2, [(0, 1, 2)], 0, 1, 3) == 3


def test_choose_path_with_timing() -> None:
    edges = [(0, 1, 0), (1, 3, 3), (0, 2, 0), (2, 3, 1)]
    assert solve(4, edges, 0, 3, 4) == 2


def test_state_revisit_with_different_mod() -> None:
    edges = [(0, 1, 0), (1, 0, 1), (1, 2, 0)]
    assert solve(3, edges, 0, 2, 2) == 3

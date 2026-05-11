from solution import solve


def test_overlapping_intervals_must_merge() -> None:
    bits = [1, 0, 1]
    assert solve(bits, [(0, 1), (1, 2)]) == 0


def test_touching_intervals_must_merge() -> None:
    bits = [1, 0, 1, 0]
    assert solve(bits, [(0, 0), (1, 1), (2, 2), (3, 3)]) == 0


def test_unsorted_intervals() -> None:
    bits = [0, 1, 1, 0, 1, 0]
    assert solve(bits, [(4, 5), (0, 0), (1, 3)]) == 1

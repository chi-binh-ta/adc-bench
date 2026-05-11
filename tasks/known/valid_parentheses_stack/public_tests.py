from solution import solve


def test_simple_valid() -> None:
    assert solve("([]{})") is True


def test_simple_invalid() -> None:
    assert solve("(]") is False


def test_empty_is_valid() -> None:
    assert solve("") is True

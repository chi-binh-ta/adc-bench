from solution import solve


def test_long_nested_string() -> None:
    s = "({[" * 20000 + "]})" * 20000
    assert solve(s) is True

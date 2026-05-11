from solution import solve


def test_order_counterexample() -> None:
    assert solve("([)]") is False


def test_closing_before_opening() -> None:
    assert solve(")(") is False


def test_deep_mixed_nesting() -> None:
    assert solve("{[()()]([])}") is True

from app.metrics import _gini


def test_gini_uniform_distribution_is_zero():
    assert _gini([10, 10, 10, 10]) == 0.0


def test_gini_empty_is_zero():
    assert _gini([]) == 0.0


def test_gini_all_zero_is_zero():
    assert _gini([0, 0, 0]) == 0.0


def test_gini_concentrated_distribution_is_high():
    assert _gini([0, 0, 0, 0, 100]) > 0.7


def test_gini_increases_with_concentration():
    uniform = _gini([25, 25, 25, 25])
    skewed = _gini([1, 1, 1, 97])
    assert skewed > uniform

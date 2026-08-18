import math
from src.generation.generator import sigmoid


def test_sigmoid_at_zero_is_half():
    assert math.isclose(sigmoid(0.0), 0.5, abs_tol=1e-9)


def test_sigmoid_large_positive_approaches_one():
    assert sigmoid(60) == 1.0
    assert sigmoid(10) > 0.999


def test_sigmoid_large_negative_approaches_zero():
    assert sigmoid(-1000) == 0.0
    assert sigmoid(-10) < 0.001


def test_sigmoid_monotonic():
    xs = [-5, -1, 0, 1, 5]
    ys = [sigmoid(x) for x in xs]
    assert ys == sorted(ys)


def test_sigmoid_never_exceeds_bounds():
    for x in [-1000, -100, -1, 0, 1, 100, 1000]:
        y = sigmoid(x)
        assert 0.0 <= y <= 1.0

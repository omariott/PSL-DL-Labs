import numpy as np

from minitorch.utils import clip, mean_squared_error


def test_clip_basic():
    x = np.array([-5, 0, 5, 10, 15])
    result = clip(x, 0, 10)
    expected = np.array([0, 0, 5, 10, 10])
    assert np.array_equal(result, expected)


def test_clip_no_change_needed():
    x = np.array([1, 2, 3])
    result = clip(x, 0, 10)
    assert np.array_equal(result, x)


def test_mean_squared_error_basic():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.5, 2.5])
    mse = mean_squared_error(pred, target)
    assert np.isclose(mse, 0.25)


def test_mean_squared_error_zero_when_equal():
    pred = np.array([1.0, 2.0, 3.0])
    mse = mean_squared_error(pred, pred)
    assert np.isclose(mse, 0.0)

import numpy as np

from minitorch.init import kaiming_normal_, normal_
from minitorch.module import Parameter


def test_normal_default_is_standard_normal():
    np.random.seed(0)
    p = Parameter(np.zeros((300, 200)))
    normal_(p)
    assert abs(p.data.mean()) < 0.02 and abs(p.data.std() - 1) < 0.02


def test_normal_mean_and_std():
    np.random.seed(0)
    p = Parameter(np.zeros((300, 200)))
    normal_(p, mean=1.0, std=0.5)
    assert abs(p.data.mean() - 1.0) < 0.01 and abs(p.data.std() - 0.5) < 0.01


def test_normal_is_in_place():
    p = Parameter(np.zeros((5, 4)))
    array = p.data
    assert normal_(p, std=0.1) is None
    assert p.data is array and np.any(array != 0)


def test_kaiming_std_uses_number_of_inputs():
    np.random.seed(0)
    p = Parameter(np.zeros((64, 1024)))        # fan_in = 1024, not 64
    kaiming_normal_(p)
    assert abs(p.data.mean()) < 0.01
    assert abs(p.data.std() - np.sqrt(2 / 1024)) < 0.03 * np.sqrt(2 / 1024)


def test_kaiming_is_in_place():
    p = Parameter(np.zeros((8, 16)))
    array = p.data
    assert kaiming_normal_(p) is None
    assert p.data is array and np.any(array != 0)

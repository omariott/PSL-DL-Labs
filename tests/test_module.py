import copy
import pickle

import numpy as np

from minitorch.layer import Linear, Residual, Sequential
from minitorch.module import Module, Parameter


def test_parameter_forwards_array_attributes():
    p = Parameter(np.zeros((3, 4)))
    assert p.shape == (3, 4) and p.ndim == 2 and p.T.shape == (4, 3)


def test_parameter_works_in_array_arithmetic():
    p = Parameter(np.ones((3, 4)))
    x = np.ones((2, 3))
    assert (x @ p).shape == (2, 4)                     # ndarray @ Parameter
    assert (np.ones((2, 4)) + Parameter(np.ones(4))).shape == (2, 4)    # ndarray + Parameter


def test_parameter_deepcopy_is_independent_parameter():
    p = Parameter(np.ones(3))
    q = copy.deepcopy(p)
    assert isinstance(q, Parameter)
    q.data[0] = 5.0
    assert p.data[0] == 1.0


def test_parameter_pickle_roundtrip():
    p = Parameter(np.arange(3.0))
    p.grad += 2
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, Parameter)
    assert np.array_equal(q.data, p.data) and np.array_equal(q.grad, p.grad)


def test_modules_start_in_training_mode():
    assert Module().training is True

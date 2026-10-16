import numpy as np

from minitorch.layer import Linear
from minitorch.optim import SGD


def test_sgd_holds_the_parameters_of_the_model():
    layer = Linear(3, 2)
    optimizer = SGD(layer.parameters(), lr=0.1)
    assert optimizer.parameters[0] is layer.weight
    assert optimizer.parameters[1] is layer.bias


def test_sgd_step_update():
    layer = Linear(3, 2)
    optimizer = SGD(layer.parameters(), lr=0.1)
    w_before = layer.weight.data.copy()
    b_before = layer.bias.data.copy()
    layer.weight.grad[...] = 1.0
    layer.bias.grad[...] = -2.0
    optimizer.step()
    assert np.allclose(layer.weight.data, w_before - 0.1)
    assert np.allclose(layer.bias.data, b_before + 0.2)


def test_sgd_step_is_seen_by_the_layer():
    layer = Linear(3, 2)
    optimizer = SGD(layer.parameters(), lr=0.5)
    x = np.random.default_rng(0).normal(size=(4, 3))
    layer.weight.grad[...] = 1.0
    layer.bias.grad[...] = 1.0
    optimizer.step()
    assert np.allclose(layer(x), x @ layer.weight.data.T + layer.bias.data)


def test_sgd_zero_grad():
    layer = Linear(3, 2)
    optimizer = SGD(layer.parameters(), lr=0.1)
    layer.weight.grad[...] = 1.0
    layer.bias.grad[...] = -2.0
    optimizer.zero_grad()
    assert np.all(layer.weight.grad == 0)
    assert np.all(layer.bias.grad == 0)

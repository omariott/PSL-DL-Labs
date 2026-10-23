import numpy as np

from minitorch.layer import Linear
from minitorch.optim import SGD, RMSProp, Adam


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


# --- SGD with momentum -----------------------------------------------------

def _set_grads(layer, w_grad, b_grad):
    layer.weight.grad[...] = w_grad
    layer.bias.grad[...] = b_grad


def test_sgd_momentum_zero_is_plain_sgd():
    np.random.seed(0)
    a = Linear(3, 2)
    np.random.seed(0)
    b = Linear(3, 2)
    opt_a = SGD(a.parameters(), lr=0.1)
    opt_b = SGD(b.parameters(), lr=0.1, momentum=0.0)
    for g in [1.0, -0.5, 2.0]:
        _set_grads(a, g, g)
        _set_grads(b, g, g)
        opt_a.step()
        opt_b.step()
    assert np.allclose(a.weight.data, b.weight.data)
    assert np.allclose(a.bias.data, b.bias.data)


def test_sgd_momentum_steps():
    layer = Linear(3, 2)
    optimizer = SGD(layer.parameters(), lr=0.1, momentum=0.9)
    w0 = layer.weight.data.copy()
    _set_grads(layer, 1.0, 1.0)
    optimizer.step()                              # v = 1, moves by -0.1
    assert np.allclose(layer.weight.data, w0 - 0.1)
    optimizer.step()                              # v = 0.9 + 1 = 1.9, moves by -0.19
    assert np.allclose(layer.weight.data, w0 - 0.1 - 0.19)
    _set_grads(layer, 0.0, 0.0)
    optimizer.step()                              # v = 1.71, keeps moving without gradient
    assert np.allclose(layer.weight.data, w0 - 0.1 - 0.19 - 0.171)


def test_sgd_momentum_keeps_one_velocity_per_parameter():
    layer = Linear(3, 2)
    optimizer = SGD(layer.parameters(), lr=0.1, momentum=0.9)
    w0, b0 = layer.weight.data.copy(), layer.bias.data.copy()
    _set_grads(layer, 1.0, -1.0)
    optimizer.step()
    optimizer.step()
    assert np.allclose(layer.weight.data, w0 - 0.29)
    assert np.allclose(layer.bias.data, b0 + 0.29)


# --- RMSProp ---------------------------------------------------------------

def test_rmsprop_holds_the_parameters_of_the_model():
    layer = Linear(3, 2)
    optimizer = RMSProp(layer.parameters(), lr=0.1)
    assert optimizer.parameters[0] is layer.weight
    assert optimizer.parameters[1] is layer.bias


def test_rmsprop_steps():
    layer = Linear(3, 2)
    optimizer = RMSProp(layer.parameters(), lr=0.1, alpha=0.9, eps=0.0)
    w0 = layer.weight.data.copy()
    layer.weight.grad[...] = np.array([[2.0, -4.0, 0.5], [1.0, 1.0, 1.0]])
    optimizer.step()
    # s = 0.1 * g ** 2, so the step is lr * g / (sqrt(0.1) * |g|) = lr * sign(g) / sqrt(0.1)
    expected = w0 - 0.1 * np.sign(layer.weight.grad) / np.sqrt(0.1)
    assert np.allclose(layer.weight.data, expected)
    optimizer.step()
    # s = 0.9 * 0.1 g^2 + 0.1 g^2 = 0.19 g^2
    expected = expected - 0.1 * np.sign(layer.weight.grad) / np.sqrt(0.19)
    assert np.allclose(layer.weight.data, expected)


def test_rmsprop_zero_grad():
    layer = Linear(3, 2)
    optimizer = RMSProp(layer.parameters(), lr=0.1)
    _set_grads(layer, 1.0, -2.0)
    optimizer.zero_grad()
    assert np.all(layer.weight.grad == 0)
    assert np.all(layer.bias.grad == 0)


# --- Adam ------------------------------------------------------------------

def test_adam_holds_the_parameters_of_the_model():
    layer = Linear(3, 2)
    optimizer = Adam(layer.parameters(), lr=0.1)
    assert optimizer.parameters[0] is layer.weight
    assert optimizer.parameters[1] is layer.bias


def test_adam_first_step_has_size_lr():
    # with bias correction, m_hat = g and v_hat = g ** 2 at the first step
    layer = Linear(3, 2)
    optimizer = Adam(layer.parameters(), lr=0.01, eps=0.0)
    w0, b0 = layer.weight.data.copy(), layer.bias.data.copy()
    layer.weight.grad[...] = np.array([[2.0, -4.0, 0.5], [1e-3, 1.0, -7.0]])
    layer.bias.grad[...] = np.array([-3.0, 0.1])
    optimizer.step()
    assert np.allclose(layer.weight.data, w0 - 0.01 * np.sign(layer.weight.grad))
    assert np.allclose(layer.bias.data, b0 - 0.01 * np.sign(layer.bias.grad))


def test_adam_counts_steps():
    layer = Linear(3, 2)
    optimizer = Adam(layer.parameters())
    assert optimizer.t == 0
    optimizer.step()
    optimizer.step()
    assert optimizer.t == 2


def test_adam_steps_match_the_formulas():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2)
    lr, b1, b2, eps = 0.05, 0.8, 0.95, 1e-8
    optimizer = Adam(layer.parameters(), lr=lr, betas=(b1, b2), eps=eps)
    w = layer.weight.data.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for t in range(1, 6):
        g = rng.normal(size=w.shape)
        layer.weight.grad[...] = g
        optimizer.step()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        w = w - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        assert np.allclose(layer.weight.data, w), f"wrong value after step {t}"


def test_adam_step_is_seen_by_the_layer():
    layer = Linear(3, 2)
    optimizer = Adam(layer.parameters(), lr=0.5)
    x = np.random.default_rng(0).normal(size=(4, 3))
    _set_grads(layer, 1.0, 1.0)
    optimizer.step()
    assert np.allclose(layer(x), x @ layer.weight.data.T + layer.bias.data)


def test_adam_zero_grad():
    layer = Linear(3, 2)
    optimizer = Adam(layer.parameters(), lr=0.1)
    _set_grads(layer, 1.0, -2.0)
    optimizer.zero_grad()
    assert np.all(layer.weight.grad == 0)
    assert np.all(layer.bias.grad == 0)

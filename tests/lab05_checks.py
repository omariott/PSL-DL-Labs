"""Checks for the functions written inside the Lab 5 notebook.

pytest cannot see functions defined in notebook cells, so each check takes the student's
functions as arguments. The notebook calls them, for example:

    checks.sigmoid(sigmoid_forward, sigmoid_backward)

A check raises an AssertionError that says what is wrong, and prints a line when everything passes.
The functions are named check-style, not test_*, so pytest does not collect them.
"""

import numpy as np

from minitorch.layer import Linear, Sigmoid, Sequential
from minitorch.loss import MSELoss
from minitorch.optim import SGD
from minitorch.utils import numerical_gradient

ATOL = 1e-6


def _pair(result, name, first, second):
    """Check that a function returned a (first, second) tuple, and return it."""
    assert isinstance(result, tuple) and len(result) == 2, f"{name} must return ({first}, {second})"
    return result


def _array(value, shape, name):
    """Check that value is an array of the given shape."""
    assert value is not None, f"{name} returned None"
    value = np.asarray(value)
    assert value.shape == tuple(shape), f"{name} has shape {value.shape}, expected {tuple(shape)}"
    return value


def _match(analytic, numerical, name):
    assert np.allclose(analytic, numerical, atol=ATOL), (
        f"{name} does not match the numerical gradient "
        f"(largest difference {np.abs(analytic - numerical).max():.2e})"
    )


def mse(mse_forward, mse_backward):
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(5, 2))
    target = rng.normal(size=(5, 2))

    loss, cache = _pair(mse_forward(pred, target), "mse_forward", "loss", "cache")
    assert np.ndim(loss) == 0, "mse_forward: the loss must be a scalar"
    assert np.isclose(loss, np.mean((pred - target) ** 2)), "mse_forward: wrong loss value"

    dpred = _array(mse_backward(cache), pred.shape, "mse_backward")
    numerical = numerical_gradient(lambda p: mse_forward(p, target)[0], pred)
    _match(dpred, numerical, "mse_backward")
    print("mse: all checks passed")


def sigmoid(sigmoid_forward, sigmoid_backward):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 3))
    g = rng.normal(size=(5, 3))

    a, cache = _pair(sigmoid_forward(z), "sigmoid_forward", "a", "cache")
    a = _array(a, z.shape, "sigmoid_forward output")
    assert np.allclose(a, 1 / (1 + np.exp(-z))), "sigmoid_forward: wrong output values"

    dz = _array(sigmoid_backward(g, cache), z.shape, "sigmoid_backward")
    numerical = numerical_gradient(lambda z_: np.sum(sigmoid_forward(z_)[0] * g), z)
    _match(dz, numerical, "sigmoid_backward")
    print("sigmoid: all checks passed")


def linear(linear_forward, linear_backward):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3))
    W = rng.normal(size=(2, 3))
    b = rng.normal(size=2)
    g = rng.normal(size=(5, 2))

    z, cache = _pair(linear_forward(x, W, b), "linear_forward", "z", "cache")
    z = _array(z, (5, 2), "linear_forward output")
    assert np.allclose(z, x @ W.T + b), "linear_forward: wrong output values"

    grads = linear_backward(g, cache)
    assert isinstance(grads, tuple) and len(grads) == 3, "linear_backward must return (dx, dW, db)"
    dx = _array(grads[0], x.shape, "linear_backward dx")
    dW = _array(grads[1], W.shape, "linear_backward dW")
    db = _array(grads[2], b.shape, "linear_backward db")

    def scalar(x_, W_, b_):
        return np.sum(linear_forward(x_, W_, b_)[0] * g)

    _match(dx, numerical_gradient(lambda x_: scalar(x_, W, b), x), "linear_backward dx")
    _match(dW, numerical_gradient(lambda W_: scalar(x, W_, b), W), "linear_backward dW")
    _match(db, numerical_gradient(lambda b_: scalar(x, W, b_), b), "linear_backward db")
    print("linear: all checks passed")


def neuron_loss_and_grads(neuron_loss_and_grads, X, y):
    rng = np.random.default_rng(0)
    W = rng.normal(size=(1, 2))
    b = rng.normal(size=1)

    result = neuron_loss_and_grads(W, b, X, y)
    assert isinstance(result, tuple) and len(result) == 3, "neuron_loss_and_grads must return (loss, dW, db)"
    loss, dW, db = result
    expected = np.mean((1 / (1 + np.exp(-(X @ W.T + b))) - y) ** 2)
    assert np.isclose(loss, expected), "neuron_loss_and_grads: wrong loss value"
    dW = _array(dW, W.shape, "neuron_loss_and_grads dW")
    db = _array(db, b.shape, "neuron_loss_and_grads db")
    _match(dW, numerical_gradient(lambda W_: neuron_loss_and_grads(W_, b, X, y)[0], W), "dW")
    _match(db, numerical_gradient(lambda b_: neuron_loss_and_grads(W, b_, X, y)[0], b), "db")
    print("neuron_loss_and_grads: all checks passed")


def train_neuron(train_neuron, X, y, W0, b0, lr, n_steps):
    W0_copy, b0_copy = W0.copy(), b0.copy()
    result = train_neuron(X, y, W0, b0, lr, n_steps)
    assert isinstance(result, tuple) and len(result) == 3, "train_neuron must return (W, b, losses)"
    W, b, losses = result
    assert np.array_equal(W0, W0_copy) and np.array_equal(b0, b0_copy), "train_neuron modified W0 or b0"
    assert len(losses) == n_steps, f"train_neuron returned {len(losses)} losses, expected {n_steps}"
    assert losses[-1] < losses[0], "the loss did not decrease during training"

    _, _, losses_frozen = train_neuron(X, y, W0, b0, 0.0, 5)
    assert np.allclose(losses_frozen, losses_frozen[0]), "with lr=0 the loss must not change"
    print("train_neuron: all checks passed")


def modules_match_functions(train, X, y, W0, b0, lr, n_steps, expected):
    """Train the Part 1 neuron with modules and compare it with the Part 1 run.

    expected is (W, b, losses), obtained in Part 1 from the same initial parameters.
    """
    W_expected, b_expected, losses_expected = expected
    model = Sequential(Linear(2, 1), Sigmoid())
    model.layers[0].weight.data[...] = W0
    model.layers[0].bias.data[...] = b0

    losses = train(model, MSELoss(), SGD(model.parameters(), lr=lr), X, y, n_steps)
    assert losses is not None and len(losses) == n_steps, f"train must return a list of {n_steps} losses"
    assert np.allclose(losses, losses_expected), "the loss curve differs from the one of Part 1"
    assert np.allclose(model.layers[0].weight.data, W_expected), "the trained weight differs from Part 1"
    assert np.allclose(model.layers[0].bias.data, b_expected), "the trained bias differs from Part 1"
    print("modules match the Part 1 functions: all checks passed")


def xor_zero_init(model_zero, losses_zero, X_xor, y_xor, n_steps):
    assert len(losses_zero) == n_steps, f"losses_zero has {len(losses_zero)} entries, expected {n_steps}"
    hidden, output = model_zero.layers[0], model_zero.layers[2]
    assert np.allclose(hidden.weight.data, hidden.weight.data[0]), "the hidden units have different weights"
    assert np.allclose(hidden.bias.data, hidden.bias.data[0]), "the hidden units have different biases"
    assert np.allclose(output.weight.data, output.weight.data[0, 0]), "the output weights are different"
    pred = model_zero(X_xor)
    assert np.sum((pred > 0.5) == (y_xor > 0.5)) <= 3, "this network cannot fit XOR, but all points are correct"
    print("zero initialization: all checks passed")


def xor_default_init(model_xor, losses_xor, X_xor, y_xor, n_steps):
    assert len(losses_xor) == n_steps, f"losses_xor has {len(losses_xor)} entries, expected {n_steps}"
    assert losses_xor[-1] < losses_xor[0], "the loss did not decrease during training"
    pred = model_xor(X_xor)
    assert np.all((pred > 0.5) == (y_xor > 0.5)), "the network does not classify all four XOR points correctly"
    print("default initialization: all checks passed")

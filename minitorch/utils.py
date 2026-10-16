"""
minitorch.utils
================
"""

import numpy as np



def numerical_gradient(f, x, h=1e-5):
    """Estimate the gradient of f at x numerically, using central differences.

    f : function that takes an array of the same shape as x and returns a single number
    x : array of any shape (a vector, a matrix, ...)
    h : size of the nudge applied to each entry of x

    Returns an array with the same shape as x. Its i-th entry is the estimated partial derivative
    of f with respect to the i-th entry of x: how fast f changes when only that entry changes.

    f is called twice per entry of x, so this is a tool for checking hand-derived gradients on
    small examples, not for training.
    """
    # Convert x to a NumPy array of floats: a nudge as small as h would be lost on integers
    x = np.asarray(x, dtype=float)

    # The result has one entry per entry of x
    grad = np.zeros_like(x)

    # x.flat gives access to the entries of x one by one, whatever the shape of x:
    # x.flat[i] is the i-th entry
    for i in range(x.size):
        # Work on copies, so that the array passed by the caller is never modified
        x_plus = x.copy()
        x_minus = x.copy()

        # Nudge only the i-th entry, forward by h in one copy and backward by h in the other.
        # One entry at a time: nudging several together would only measure their combined effect.
        x_plus.flat[i] += h
        x_minus.flat[i] -= h

        # Central difference: how much f changed over a total move of 2h along entry i
        # (the same formula as numerical_derivative, applied to one entry at a time)
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h)

    return grad


def _compare(analytic, numerical, name, atol):
    """Raise an AssertionError naming the gradient if it does not match."""
    assert analytic.shape == numerical.shape, (
        f"gradient of {name} has shape {analytic.shape}, expected {numerical.shape}"
    )
    assert np.allclose(analytic, numerical, atol=atol), (
        f"gradient of {name} does not match the numerical gradient "
        f"(largest difference {np.abs(analytic - numerical).max():.2e})"
    )


def _numerical_parameter_gradient(f, p):
    """Numerical gradient of f() with respect to the values of the Parameter p."""
    original = p.data.copy()

    def f_of_values(values):
        p.data[...] = values
        return f()

    grad = numerical_gradient(f_of_values, original)
    p.data[...] = original
    return grad


def check_module(module, x, atol=1e-6):
    """Check the backward of a module against numerical gradients.

    The output of the module is turned into a scalar by multiplying it elementwise with a
    fixed random array g and summing. The same g is used as the upstream gradient, so
    backward must return the numerical gradient of that scalar with respect to the input,
    and fill in the .grad of every parameter with its numerical gradient.
    Use float64 arrays. Raises an AssertionError that names the first gradient that is wrong.
    """
    rng = np.random.default_rng(0)
    g = rng.normal(size=module(x).shape)

    module.zero_grad()
    module(x)
    dx = module.backward(g)
    param_grads = [p.grad.copy() for p in module.parameters()]
    module.zero_grad()

    def scalar():
        return np.sum(module(x) * g)

    _compare(dx, numerical_gradient(lambda x_: np.sum(module(x_) * g), x), "the input", atol)
    for i, p in enumerate(module.parameters()):
        _compare(param_grads[i], _numerical_parameter_gradient(scalar, p), f"parameter {i}", atol)


def check_loss(loss_fn, pred, target, atol=1e-6):
    """Check the backward of a loss against the numerical gradient with respect to pred."""
    loss_fn(pred, target)
    dpred = loss_fn.backward()
    numerical = numerical_gradient(lambda p_: loss_fn(p_, target), pred)
    _compare(dpred, numerical, "pred", atol)


def check_model(model, loss_fn, x, y, atol=1e-6):
    """Check the gradients of a whole model, loss included, against numerical gradients.

    The analytic gradients come from loss_fn.backward() followed by model.backward().
    They are compared with the numerical gradient of the loss with respect to the input
    and to every parameter. Raises an AssertionError that names the first gradient that is wrong.
    """
    model.zero_grad()
    loss_fn(model(x), y)
    dx = model.backward(loss_fn.backward())
    param_grads = [p.grad.copy() for p in model.parameters()]
    model.zero_grad()

    def loss():
        return loss_fn(model(x), y)

    _compare(dx, numerical_gradient(lambda x_: loss_fn(model(x_), y), x), "the input", atol)
    for i, p in enumerate(model.parameters()):
        _compare(param_grads[i], _numerical_parameter_gradient(loss, p), f"parameter {i}", atol)



def clip(x, lo, hi):
    """Clip the values in `x` to the range [lo, hi].

    Parameters
    ----------
    x : np.ndarray
        Input array.
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Returns
    -------
    np.ndarray
        Array with values clipped to [lo, hi].

    Examples
    --------
    >>> clip(np.array([-5, 0, 5, 10, 15]), 0, 10)
    array([ 0,  0,  5, 10, 10])
    """
    # TODO: implement this function.
    # Hint: look up np.clip, or implement it yourself with
    # np.minimum / np.maximum.
    return np.clip(x, lo, hi) 


def mean_squared_error(pred, target):
    """Compute the mean squared error between predictions and targets.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values.
    target : np.ndarray
        Ground-truth values, same shape as `pred`.

    Returns
    -------
    float
        The mean squared error between `pred` and `target`.

    Examples
    --------
    >>> mean_squared_error(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 2.5]))
    0.25
    """
    # TODO: implement this function.
    # Hint: this is the average of the squared differences between
    # pred and target. No loops needed.
    return np.mean((pred - target) ** 2)
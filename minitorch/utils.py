"""
minitorch.utils
================

Small utility functions used throughout the semester's library.

For Lab 1, you will implement two simple functions:
    - clip(x, lo, hi)
    - mean_squared_error(pred, target)

Both should be implemented using NumPy only (no Python for-loops).
This is meant to be a gentle first contact with the codebase, not
a hard exercise -- if you're stuck for more than a few minutes,
ask for help!
"""

import numpy as np


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

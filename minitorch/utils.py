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
    raise NotImplementedError("clip() is not implemented yet")


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
    raise NotImplementedError("mean_squared_error() is not implemented yet")

"""Loss functions."""

import numpy as np

from .module import Module


class MSELoss(Module):
    """Mean squared error, averaged over all elements of pred.

    A loss is the last module of the chain: its backward takes no argument and
    returns the gradient of the loss with respect to pred.
    """

    def __init__(self):
        self.cache = None

    def forward(self, pred, target):
        """Mean of the squared differences between pred and target. Returns a scalar.

        Saves what backward needs in self.cache.
        """
        ...

    def backward(self):
        """Gradient of the loss with respect to pred, with the shape of pred."""
        ...


class BCELoss(Module):
    """Binary cross-entropy, averaged over all elements of pred.

    pred holds probabilities in (0, 1), usually the output of a Sigmoid, and target holds
    labels in {0, 1}. pred is clipped to [eps, 1 - eps] so that the logarithms stay finite.
    """

    def __init__(self, eps=1e-7):
        self.eps = eps
        self.cache = None

    def forward(self, pred, target):
        """Mean of -(target * log(pred) + (1 - target) * log(1 - pred)). Returns a scalar.

        Saves what backward needs in self.cache.
        """
        ...

    def backward(self):
        """Gradient of the loss with respect to pred, with the shape of pred."""
        ...

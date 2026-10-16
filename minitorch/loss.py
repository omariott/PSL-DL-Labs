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

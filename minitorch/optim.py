"""Optimizers."""

import numpy as np


class SGD:
    """Stochastic gradient descent: every parameter moves against its gradient.

    parameters: the list returned by model.parameters(). It holds the Parameter objects
    the layers use, not copies, so an update made here is seen by the next forward pass.
    """

    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        """Update every parameter in place."""
        ...

    def zero_grad(self):
        """Set the gradient of every parameter to zero (provided)."""
        for p in self.parameters:
            p.grad.fill(0)

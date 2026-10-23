"""Optimizers."""

import numpy as np


class SGD:
    """Stochastic gradient descent: every parameter moves against its gradient.

    parameters: the list returned by model.parameters(). It holds the Parameter objects
    the layers use, not copies, so an update made here is seen by the next forward pass.

    With momentum > 0, each parameter keeps a velocity v, an array of its shape that starts
    at zero. At every step, v = momentum * v + grad and the parameter moves by -lr * v.
    With momentum = 0 this is plain gradient descent, as in Lab 5.
    """

    def __init__(self, parameters, lr, momentum=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Update every parameter in place."""
        ...

    def zero_grad(self):
        """Set the gradient of every parameter to zero (provided)."""
        for p in self.parameters:
            p.grad.fill(0)


class RMSProp:
    """RMSProp: every entry of every parameter gets its own step size.

    Each parameter keeps a running average s of its squared gradient, an array of its shape
    that starts at zero. 

    All operations are elementwise.
    """

    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.squares = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Update every parameter in place."""
        ...

    def zero_grad(self):
        """Set the gradient of every parameter to zero (provided)."""
        for p in self.parameters:
            p.grad.fill(0)


class Adam:
    """Adam: momentum and RMSProp together, with a bias correction.

    Each parameter keeps a running average m of its gradient and a running average v of its
    squared gradient, two arrays of its shape that start at zero. self.t counts the steps
    taken, starting at 0. 

    All operations are elementwise. betas is the pair (beta1, beta2).
    """

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.99), eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.means = [np.zeros_like(p.data) for p in self.parameters]
        self.squares = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Update every parameter in place."""
        ...

    def zero_grad(self):
        """Set the gradient of every parameter to zero (provided)."""
        for p in self.parameters:
            p.grad.fill(0)

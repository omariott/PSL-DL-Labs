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
        for p, v in zip(self.parameters, self.velocities):
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v

    def zero_grad(self):
        """Set the gradient of every parameter to zero (provided)."""
        for p in self.parameters:
            p.grad.fill(0)


class RMSProp:
    """RMSProp: every entry of every parameter gets its own step size.

    Each parameter keeps a running average s of its squared gradient, an array of its shape
    that starts at zero. At every step,

        s = alpha * s + (1 - alpha) * grad ** 2
        parameter -= lr * grad / (sqrt(s) + eps)

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
        for p, s in zip(self.parameters, self.squares):
            s *= self.alpha
            s += (1 - self.alpha) * p.grad ** 2
            p.data -= self.lr * p.grad / (np.sqrt(s) + self.eps)

    def zero_grad(self):
        """Set the gradient of every parameter to zero (provided)."""
        for p in self.parameters:
            p.grad.fill(0)


class Adam:
    """Adam: momentum and RMSProp together, with a bias correction.

    Each parameter keeps a running average m of its gradient and a running average v of its
    squared gradient, two arrays of its shape that start at zero. self.t counts the steps
    taken, starting at 0. At every step, first t is increased by 1, then

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        parameter -= lr * m_hat / (sqrt(v_hat) + eps)

    All operations are elementwise. betas is the pair (beta1, beta2).
    """

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.means = [np.zeros_like(p.data) for p in self.parameters]
        self.squares = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Update every parameter in place."""
        self.t += 1
        for p, m, v in zip(self.parameters, self.means, self.squares):
            m *= self.beta1
            m += (1 - self.beta1) * p.grad
            v *= self.beta2
            v += (1 - self.beta2) * p.grad ** 2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Set the gradient of every parameter to zero (provided)."""
        for p in self.parameters:
            p.grad.fill(0)

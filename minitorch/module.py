"""Building blocks shared by every layer (provided, not edited by students)."""

import numpy as np


class Parameter:
    """A learnable array together with its gradient.

    data: the current values, a NumPy array.
    grad: the gradient of the loss with respect to data, an array of the same shape.
          backward fills it in and the optimizer reads it.
    """

    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)
        
    def __getattr__(self, name):
        return getattr(self.data, name)


class Module:
    """Base class of every building block of a network.

    A module implements two methods:

    forward(x)         computes the output and saves what backward will need.
    backward(grad)     receives the gradient of the loss with respect to the module's output.
                       It adds the gradients of the module's parameters to their .grad
                       and returns the gradient with respect to the module's input.

    Calling a module, as in model(x), runs forward.
    """

    def __call__(self, *args):
        return self.forward(*args)

    def parameters(self):
        """The module's Parameter objects, as a list. Modules without parameters return []."""
        return []

    def zero_grad(self):
        """Set the gradient of every parameter to zero.

        backward adds to .grad instead of replacing it, so the gradients must be reset
        before each new backward pass.
        """
        for p in self.parameters():
            p.grad.fill(0)

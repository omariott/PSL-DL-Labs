"""Building blocks shared by every layer (provided, not edited by students)."""

import numpy as np


class Buffer:
    """An array that belongs to the state of a module but is not learned: it has no gradient.

    data: the current values, a NumPy array. Update it in place (buffer.data[...] = ...).
    Modules list their buffers in buffers(); an example is the running statistics of batch normalization.
    """

    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        # Only called when the normal lookup fails: attributes such as shape, ndim or T are
        # forwarded to the array, and so are the NumPy conversion hooks that let arithmetic like
        # `x + parameter` or `grad @ parameter` work. The copy and pickle hooks and 'data' itself
        # are not forwarded, otherwise copy.deepcopy returns a bare array and unpickling recurses.
        if name in _NOT_FORWARDED:
            raise AttributeError(name)
        return getattr(self.data, name)


_NOT_FORWARDED = {"data", "__deepcopy__", "__copy__", "__getstate__", "__setstate__"}


class Parameter(Buffer):
    """A learnable array together with its gradient (a Buffer that is also learned).

    data: the current values, a NumPy array.
    grad: the gradient of the loss with respect to data, an array of the same shape.
          backward fills it in and the optimizer reads it.
    """

    def __init__(self, data):
        super().__init__(data)
        self.grad = np.zeros_like(data)


class Module:
    """Base class of every building block of a network.

    A module implements two methods:

    forward(x)         computes the output and saves what backward will need.
    backward(grad)     receives the gradient of the loss with respect to the module's output.
                       It adds the gradients of the module's parameters to their .grad
                       and returns the gradient with respect to the module's input.

    Calling a module, as in model(x), runs forward.

    Modules have a mode: model.train() and model.eval() set the flag self.training of the
    module and of all the modules it contains. Layers that behave differently in the two
    modes (BatchNorm1d) read self.training.
    """

    # Class attribute: layers written without calling a base-class __init__ are in training mode.
    training = True

    def __call__(self, *args):
        return self.forward(*args)

    def parameters(self):
        """The module's Parameter objects, as a list. Modules without parameters return []."""
        return []

    def buffers(self):
        """The module's Buffer objects, as a list: arrays of its state that are not learned.

        Containers return those of the modules they contain; modules with buffers of their own override it.
        """
        return [b for child in self.children() for b in child.buffers()]

    def children(self):
        """The modules directly contained in this module, as a list. Containers override it."""
        return []

    def train(self, mode=True):
        """Set the mode of this module and of every module it contains. Returns the module."""
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self):
        """Switch this module and every module it contains to eval mode."""
        return self.train(False)

    def zero_grad(self):
        """Set the gradient of every parameter to zero.

        backward adds to .grad instead of replacing it, so the gradients must be reset
        before each new backward pass.
        """
        for p in self.parameters():
            p.grad.fill(0)

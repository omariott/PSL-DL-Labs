"""
minitorch.layer
"""

import numpy as np

from .module import Module, Parameter


class Linear(Module):
    """Fully connected layer: a batch of inputs x of shape (N, in_features) is mapped
    to x @ W.T + b, of shape (N, out_features).

    weight has shape (out_features, in_features) and bias has shape (out_features,),
    both Parameter objects. Only 2D inputs are supported.
    """

    def __init__(self, in_features, out_features):
        # Drawn from a standard normal distribution (Lab 6 covers initialization schemes).
        self.weight = Parameter(np.random.randn(out_features, in_features))
        self.bias = Parameter(np.zeros(out_features))
        self.cache = None

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        """Affine map of the batch x, of shape (N, in_features).

        Returns an array of shape (N, out_features). Saves what backward needs.
        Read weight.data and bias.data on every call, since the optimizer changes them.
        """
        ...

    def backward(self, grad):
        """Backward pass.

        grad has shape (N, out_features): the gradient of the loss with respect to the output.
        Adds the gradients of the loss with respect to weight and bias to their .grad
        (accumulate with +=, never overwrite) and returns the gradient with respect to
        the input, of shape (N, in_features).
        """
        ...


class Sigmoid(Module):
    """Elementwise sigmoid."""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """Sigmoid of x, with the shape of x. Saves what backward needs."""
        ...

    def backward(self, grad):
        """grad is the gradient of the loss with respect to the output.

        Returns the gradient with respect to the input.
        """
        ...


class Sequential(Module):
    """A chain of modules applied in order."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        """grad is the gradient of the loss with respect to the output of the model.

        Returns the gradient with respect to its input.
        """
        ...

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class Perceptron:
    """A single perceptron for binary classification with labels in {-1, +1}.

    Two methods are implemented:
        - predict(x): returns the perceptron's current prediction for one input
        - learn(x, y): updates the perceptron's parameters if predict(x) does
          not match the true label y
    """

    def __init__(self, num_features):
        """Create a perceptron for inputs with `num_features` features.

        The weight starts as all zeros, one entry per feature, and the bias
        starts at zero.
        """
        self.weight = np.zeros(num_features)
        self.bias = 0.0

    def predict(self, x):
        """Return the perceptron's prediction for a single input.

        Parameters
        ----------
        x : np.ndarray
            A single input, shape (num_features,).

        Returns
        -------
        int
            +1 or -1.
        """
        # TODO: implement this method, following the decision rule from the notebook.
        return 1 if np.dot(x, self.weight) + self.bias >= 0 else -1

    def learn(self, x, y):
        """Update the perceptron from one training example, if needed.

        Parameters
        ----------
        x : np.ndarray
            A single input, shape (num_features,).
        y : int
            The true label for x, +1 or -1.

        Returns
        -------
        bool
            True if the parameters were updated (the prediction was wrong),
            False if the prediction was already correct.
        """
        # TODO: implement this method, following the update rule from the notebook.
        pred = self.predict(x)
        if pred * y < 0:
            self.weight += y * x
            self.bias += y
            return True
        return False


class RandomProjection:
    """A fixed random projection followed by a threshold at 0.

    In the style of the association units in Rosenblatt's original Mark I
    Perceptron: a random projection followed by a threshold. self.weight is
    generated once, when the layer is created, and is never updated
    afterward.
    """

    def __init__(self, num_inputs, num_outputs):
        """Create the layer's fixed random weight.

        The weight is stored as self.weight, with shape
        (num_inputs, num_outputs), filled with random values following a
        standard normal distribution. It is generated once here and is
        never modified afterward.

        Parameters
        ----------
        num_inputs : int
            Number of features in each input.
        num_outputs : int
            Number of random features the layer produces.
        """
        # TODO: implement this method.
        # Hint: NumPy has a function that produces normally distributed samples.
        self.weight = np.random.randn(num_inputs, num_outputs)

    def transform(self, x):
        """Apply the fixed random projection and threshold to a batch of inputs.

        Parameters
        ----------
        x : np.ndarray
            A batch of inputs, shape (num_samples, num_inputs).

        Returns
        -------
        np.ndarray
            The thresholded random features, shape (num_samples, num_outputs).
            Every value is either 0.0 or 1.0.
        """
        # TODO: implement this method.
        return (np.dot(x, self.weight) >= 0).astype('float')


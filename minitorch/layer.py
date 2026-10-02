"""
minitorch.layer
"""

import numpy as np


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
        # TODO: implement this method.
        raise NotImplementedError("predict() is not implemented yet")

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
        # TODO: implement this method.
        raise NotImplementedError("learn() is not implemented yet")


class RandomProjection:
    """A fixed random projection followed by a threshold at 0.

    In the style of the association units in Rosenblatt's original Mark I
    Perceptron: a random projection followed by a threshold.
    """

    def __init__(self, num_inputs, num_outputs):
        """Create the layer's fixed random weight.

        Parameters
        ----------
        num_inputs : int
            Number of features in each input.
        num_outputs : int
            Number of random features the layer produces.
        """
        # TODO: implement this method.
        # Hint: NumPy has a function that produces normally distributed samples.
        raise NotImplementedError("__init__() is not implemented yet")

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
        raise NotImplementedError("transform() is not implemented yet")

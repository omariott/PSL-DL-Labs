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
        self.cache = 2 * (pred - target) / pred.size
        return np.mean((pred - target) ** 2)

    def backward(self):
        """Gradient of the loss with respect to pred, with the shape of pred."""
        return self.cache


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
        p = np.clip(pred, self.eps, 1 - self.eps)
        self.cache = (p - target) / (p * (1 - p) * pred.size)
        return -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))

    def backward(self):
        """Gradient of the loss with respect to pred, with the shape of pred."""
        return self.cache

class CrossEntropyLoss(Module):
    """Softmax followed by cross-entropy, averaged over the batch (as torch.nn.CrossEntropyLoss).
 
    forward takes the raw scores `logits` of shape (N, C) (no softmax before the loss) and the
    integer class labels `target` of shape (N,). The softmax is computed in a numerically stable way.
    """
 
    def __init__(self):
        self.cache = None
 
    def forward(self, logits, target):
        """Mean over the batch of -log(softmax(logits)[i, target[i]]). Returns a scalar."""
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        self.cache = (np.exp(log_probs), target)
        return -np.mean(log_probs[np.arange(len(target)), target])
 
    def backward(self):
        """Gradient of the loss with respect to logits: (softmax - one_hot(target)) / N."""
        probs, target = self.cache
        grad = probs.copy()
        grad[np.arange(len(target)), target] -= 1
        return grad / len(target)
 

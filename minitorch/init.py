"""Weight initializers. Each function overwrites param.data in place (as torch.nn.init does)."""

import numpy as np


def normal_(param, mean=0.0, std=1.0):
    """Fill param.data in place with samples from a normal distribution N(mean, std**2).

    In place: param.data must be the same array object before and after the call.
    Returns nothing.
    """
    param.data[...] = np.random.randn(*param.shape) * std + mean


def kaiming_normal_(param):
    """He initialization for ReLU networks: fill param.data in place with samples from
    N(0, 2 / fan_in).

    fan_in is the number of inputs of the layer: param.shape[1] for a weight of shape
    (out_features, in_features). Returns nothing.
    """
    ...
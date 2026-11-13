"""Transforms: callables applied to a batch of images of shape (N, H, W)."""

import numpy as np


class Compose:
    """Chain transforms: Compose([t1, t2])(x) is t2(t1(x))."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class Flatten:
    """Turn a batch of shape (N, ...) into a batch of vectors of shape (N, features)."""

    def __call__(self, x):
        return x.reshape(len(x), -1)


class Normalize:
    """Return (x - mean) / std, with mean and std scalars."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


class RandomCrop:
    """Pad every image with `padding` zeros on each side, then crop a size x size window at a
    random position, chosen independently for each image.
    """

    def __init__(self, size, padding=0):
        self.size = size
        self.padding = padding

    def __call__(self, x):
        """x has shape (N, H, W). Returns an array of shape (N, size, size) with the dtype of x.

        Use np.random for the random positions.
        """
        n, h, w = x.shape
        p = self.padding
        padded = ...
        corner_tops = ...
        corner_lefts = ...
        out = np.empty((n, self.size, self.size), dtype=x.dtype)
        ...                 #put cropped images in out
        return out
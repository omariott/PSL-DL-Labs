"""Data loading (provided)."""

from pathlib import Path

import numpy as np

EMNIST_PATH = Path(__file__).resolve().parent.parent / "data" / "emnist.npz"


def load_emnist(path=EMNIST_PATH):
    """Load MNIST from an .npz file.

    Returns ((x_train, y_train), (x_test, y_test)): images as uint8 arrays of shape (N, 28, 28)
    with values in [0, 255], labels as int64 arrays of shape (N,) with values in 0..9.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: put emnist.npz in the data/ folder of the repository.")
    with np.load(path) as f:
        return ((f["x_train"], f["y_train"].astype(np.int64)),
                (f["x_test"], f["y_test"].astype(np.int64)))


class DataLoader:
    """Iterate over (x, y) mini-batches of two arrays with the same first dimension.

    batch_size: number of samples per batch (the last batch can be smaller).
    shuffle:    draw a new random order at the start of every iteration (np.random).
    transform:  optional callable applied to every batch of x (not to y).
    """

    def __init__(self, x, y, batch_size=64, shuffle=False, transform=None):
        assert len(x) == len(y)
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.x)
        order = np.random.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            xb, yb = self.x[idx], self.y[idx]
            if self.transform is not None:
                xb = self.transform(xb)
            yield xb, yb

import numpy as np

from minitorch.loss import MSELoss
from minitorch.utils import check_loss


def test_mse_forward_value():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(5, 2))
    target = rng.normal(size=(5, 2))
    loss = MSELoss()(pred, target)
    assert np.ndim(loss) == 0
    assert np.isclose(loss, np.mean((pred - target) ** 2))


def test_mse_backward_shape():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(5, 2))
    target = rng.normal(size=(5, 2))
    loss_fn = MSELoss()
    loss_fn(pred, target)
    assert loss_fn.backward().shape == pred.shape


def test_mse_gradient():
    rng = np.random.default_rng(0)
    for shape in [(5, 2), (4, 1), (1, 3)]:
        pred = rng.normal(size=shape)
        target = rng.normal(size=shape)
        check_loss(MSELoss(), pred, target)

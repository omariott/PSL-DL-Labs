import numpy as np

from minitorch.loss import BCELoss, MSELoss
from minitorch.utils import check_loss

from minitorch.loss import CrossEntropyLoss
 
 
def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
 
 
def test_cross_entropy_value():
    np.random.seed(0)
    z, y = np.random.randn(6, 10), np.random.randint(0, 10, 6)
    expected = -np.mean(np.log(_softmax(z)[np.arange(6), y]))
    assert np.isclose(CrossEntropyLoss()(z, y), expected)
 
 
def test_cross_entropy_gradient_is_softmax_minus_onehot_over_n():
    np.random.seed(0)
    z, y = np.random.randn(6, 10), np.random.randint(0, 10, 6)
    loss = CrossEntropyLoss()
    loss(z, y)
    onehot = np.eye(10)[y]
    assert np.allclose(loss.backward(), (_softmax(z) - onehot) / 6)
 
 
def test_cross_entropy_gradient_check():
    np.random.seed(0)
    z, y = np.random.randn(4, 5), np.random.randint(0, 5, 4)
    loss = CrossEntropyLoss()
    num = np.zeros_like(z)
    for idx in np.ndindex(*z.shape):
        old = z[idx]
        z[idx] = old + 1e-6
        plus = loss(z, y)
        z[idx] = old - 1e-6
        minus = loss(z, y)
        z[idx] = old
        num[idx] = (plus - minus) / 2e-6
    loss(z, y)
    assert np.allclose(loss.backward(), num, atol=1e-7)
 
 
def test_cross_entropy_is_stable_for_large_logits():
    z = np.array([[1000.0, 0.0, -1000.0], [-1000.0, 1000.0, 0.0]])
    loss = CrossEntropyLoss()
    value = loss(z, np.array([0, 1]))
    assert np.isfinite(value) and np.isclose(value, 0.0)
    assert np.all(np.isfinite(loss.backward()))
 


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


def test_bce_forward_value():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0.05, 0.95, size=(10, 1))
    target = rng.integers(0, 2, size=(10, 1)).astype(float)
    loss = BCELoss()(pred, target)
    assert np.ndim(loss) == 0
    assert np.isclose(loss, -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred)))


def test_bce_is_finite_at_saturated_predictions():
    pred = np.array([[0.0], [1.0]])
    target = np.array([[1.0], [0.0]])
    loss_fn = BCELoss()
    assert np.isfinite(loss_fn(pred, target))
    assert np.all(np.isfinite(loss_fn.backward()))


def test_bce_backward_shape():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0.05, 0.95, size=(5, 2))
    target = rng.integers(0, 2, size=(5, 2)).astype(float)
    loss_fn = BCELoss()
    loss_fn(pred, target)
    assert loss_fn.backward().shape == pred.shape


def test_bce_gradient():
    rng = np.random.default_rng(0)
    for shape in [(5, 2), (4, 1), (1, 3)]:
        pred = rng.uniform(0.05, 0.95, size=shape)
        target = rng.integers(0, 2, size=shape).astype(float)
        check_loss(BCELoss(), pred, target)

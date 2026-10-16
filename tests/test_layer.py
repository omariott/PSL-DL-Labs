import numpy as np

from minitorch.layer import Perceptron, RandomProjection

from minitorch.layer import Linear, Sigmoid, Sequential
from minitorch.loss import MSELoss
from minitorch.utils import check_module, check_model
 
 
# --- Sigmoid ---------------------------------------------------------------
 
def test_sigmoid_forward():
    x = np.random.default_rng(0).normal(size=(5, 3))
    out = Sigmoid()(x)
    assert out.shape == x.shape
    assert np.allclose(out, 1 / (1 + np.exp(-x)))
 
 
def test_sigmoid_has_no_parameters():
    assert Sigmoid().parameters() == []
 
 
def test_sigmoid_backward_shape():
    x = np.random.default_rng(0).normal(size=(5, 3))
    sigmoid = Sigmoid()
    sigmoid(x)
    assert sigmoid.backward(np.ones((5, 3))).shape == x.shape
 
 
def test_sigmoid_gradient():
    x = np.random.default_rng(0).normal(size=(5, 3))
    check_module(Sigmoid(), x)
 
 
# --- Linear ----------------------------------------------------------------
 
def test_linear_parameter_shapes():
    layer = Linear(3, 2)
    assert layer.weight.data.shape == (2, 3)
    assert layer.bias.data.shape == (2,)
    assert layer.parameters() == [layer.weight, layer.bias]
    assert layer.weight.grad.shape == (2, 3)
    assert layer.bias.grad.shape == (2,)
 
 
def test_linear_forward_value():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2)
    layer.weight.data[...] = rng.normal(size=(2, 3))
    layer.bias.data[...] = rng.normal(size=2)
    x = rng.normal(size=(5, 3))
    out = layer(x)
    assert out.shape == (5, 2)
    assert np.allclose(out, x @ layer.weight.data.T + layer.bias.data)
 
 
def test_linear_backward_shape():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2)
    x = rng.normal(size=(5, 3))
    layer(x)
    assert layer.backward(rng.normal(size=(5, 2))).shape == x.shape
 
 
def test_linear_gradient():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2)
    layer.weight.data[...] = rng.normal(size=(2, 3))
    layer.bias.data[...] = rng.normal(size=2)
    check_module(layer, rng.normal(size=(5, 3)))
 
 
def test_linear_gradients_accumulate():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2)
    x = rng.normal(size=(5, 3))
    g = rng.normal(size=(5, 2))
    layer.zero_grad()
    layer(x)
    layer.backward(g)
    first_weight = layer.weight.grad.copy()
    first_bias = layer.bias.grad.copy()
    layer(x)
    layer.backward(g)
    assert np.allclose(layer.weight.grad, 2 * first_weight)
    assert np.allclose(layer.bias.grad, 2 * first_bias)
 
 
def test_linear_zero_grad():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2)
    layer(rng.normal(size=(5, 3)))
    layer.backward(rng.normal(size=(5, 2)))
    layer.zero_grad()
    assert np.all(layer.weight.grad == 0)
    assert np.all(layer.bias.grad == 0)
 
 
# --- Sequential ------------------------------------------------------------
 
def test_sequential_parameters():
    model = Sequential(Linear(2, 4), Sigmoid(), Linear(4, 1), Sigmoid())
    assert len(model.parameters()) == 4
 
 
def test_sequential_backward_shape():
    rng = np.random.default_rng(0)
    model = Sequential(Linear(2, 4), Sigmoid(), Linear(4, 1), Sigmoid())
    x = rng.normal(size=(6, 2))
    out = model(x)
    assert out.shape == (6, 1)
    assert model.backward(np.ones_like(out)).shape == x.shape
 
 
def test_sequential_gradient():
    rng = np.random.default_rng(0)
    np.random.seed(0)
    model = Sequential(Linear(2, 4), Sigmoid(), Linear(4, 1), Sigmoid())
    x = rng.normal(size=(6, 2))
    y = rng.integers(0, 2, size=(6, 1)).astype(float)
    check_model(model, MSELoss(), x, y)
 

# --- Perceptron ---

def test_predict_positive():
    model = Perceptron(num_features=2)
    model.weight = np.array([1.0, 1.0])
    model.bias = 0.0
    assert model.predict(np.array([1.0, 1.0])) == 1


def test_predict_negative():
    model = Perceptron(num_features=2)
    model.weight = np.array([1.0, 1.0])
    model.bias = -5.0
    assert model.predict(np.array([1.0, 1.0])) == -1


def test_predict_boundary_counts_as_positive():
    model = Perceptron(num_features=2)
    model.weight = np.array([1.0, 0.0])
    model.bias = 0.0
    assert model.predict(np.array([0.0, 5.0])) == 1  # score is exactly 0


def test_learn_updates_on_mistake():
    # a fresh, zero-initialized perceptron predicts +1 (the score is exactly
    # 0), so a true label of -1 is guaranteed to be a mistake
    model = Perceptron(num_features=2)
    x = np.array([1.0, 2.0])
    y = -1
    updated = model.learn(x, y)
    assert updated is True
    assert np.array_equal(model.weight, np.array([-1.0, -2.0]))
    assert model.bias == -1.0


def test_learn_does_not_update_on_correct_prediction():
    model = Perceptron(num_features=2)
    model.weight = np.array([1.0, 1.0])
    model.bias = 0.0
    x = np.array([1.0, 1.0])
    y = 1  # predict() should already return 1 here
    updated = model.learn(x, y)
    assert updated is False
    assert np.array_equal(model.weight, np.array([1.0, 1.0]))
    assert model.bias == 0.0


def test_learn_converges_on_separable_data():
    np.random.seed(0)
    n = 20
    pos = np.random.randn(n, 2) * 0.3 + np.array([2.0, 2.0])
    neg = np.random.randn(n, 2) * 0.3 + np.array([-2.0, -2.0])
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(n), -np.ones(n)])

    model = Perceptron(num_features=2)
    mistakes = None
    for _ in range(50):
        mistakes = 0
        for xi, yi in zip(X, y):
            if model.learn(xi, yi):
                mistakes += 1
        if mistakes == 0:
            break

    assert mistakes == 0, "the perceptron should converge to zero mistakes on clearly separable data"


# --- RandomProjection ---

def test_init_weight_shape():
    layer = RandomProjection(num_inputs=4, num_outputs=10)
    assert layer.weight.shape == (4, 10)


def test_init_weight_is_standard_normal():
    np.random.seed(0)
    weight = RandomProjection(num_inputs=500, num_outputs=500).weight
    assert abs(weight.mean()) < 0.05, "the weight values should have a mean close to 0"
    assert abs(weight.std() - 1.0) < 0.1, "the weight values should have a standard deviation close to 1"


def test_init_is_reproducible_with_seed():
    np.random.seed(0)
    first = RandomProjection(num_inputs=4, num_outputs=10).weight
    np.random.seed(0)
    second = RandomProjection(num_inputs=4, num_outputs=10).weight
    assert np.array_equal(first, second), "the weight should come from NumPy's random generator, so np.random.seed makes it reproducible"


def test_init_gives_different_weights_to_different_layers():
    first = RandomProjection(num_inputs=4, num_outputs=10).weight
    second = RandomProjection(num_inputs=4, num_outputs=10).weight
    assert not np.array_equal(first, second)


def test_transform_shape():
    layer = RandomProjection(num_inputs=4, num_outputs=10)
    x = np.random.randn(5, 4)
    out = layer.transform(x)
    assert out.shape == (5, 10)


def test_transform_thresholding():
    layer = RandomProjection(num_inputs=2, num_outputs=3)
    layer.weight = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    x = np.array([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
    expected = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    out = layer.transform(x)
    assert np.array_equal(out, expected), f"expected {expected.tolist()}, got {out.tolist()}"


def test_transform_values_are_binary():
    layer = RandomProjection(num_inputs=3, num_outputs=5)
    x = np.random.randn(10, 3)
    out = layer.transform(x)
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_weight_is_not_modified_by_transform():
    layer = RandomProjection(num_inputs=4, num_outputs=6)
    weight_before = layer.weight.copy()
    layer.transform(np.random.randn(5, 4))
    assert np.array_equal(layer.weight, weight_before), "transform() should not modify weight"

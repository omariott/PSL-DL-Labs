"""Checks for the code written inside the Lab 6 notebook.

pytest cannot see functions defined in notebook cells, so each check takes the student's
functions or results as arguments. The notebook calls them, for example:

    checks.grad_wrt_z(grad_wrt_z)

A check raises an AssertionError that says what is wrong, and prints a line when everything passes.
The functions are named check-style, not test_*, so pytest does not collect them.
"""

import numpy as np

from minitorch.loss import BCELoss, MSELoss

X_XOR = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_XOR = np.array([[0.0], [1.0], [1.0], [0.0]])


def _fits_xor(model):
    return bool(np.all((model(X_XOR) > 0.5) == (Y_XOR > 0.5)))


def grad_wrt_z(grad_wrt_z):
    rng = np.random.default_rng(0)
    z = rng.normal(scale=3.0, size=(5, 1))
    target = rng.integers(0, 2, size=(5, 1)).astype(float)
    a = 1 / (1 + np.exp(-z))

    for loss_fn, expected in [
        (MSELoss(), 2 * (a - target) * a * (1 - a) / z.size),
        (BCELoss(), (a - target) / z.size),
    ]:
        name = type(loss_fn).__name__
        dz = grad_wrt_z(loss_fn, z, target)
        assert dz is not None, f"grad_wrt_z returned None for {name}"
        dz = np.asarray(dz)
        assert dz.shape == z.shape, f"grad_wrt_z has shape {dz.shape} for {name}, expected {z.shape}"
        assert np.allclose(dz, expected, atol=1e-6), f"grad_wrt_z gives a wrong gradient for {name}"
    print("grad_wrt_z: all checks passed")


def xor_mse_vs_bce(model_mse, acc_mse, model_bce, acc_bce, n_epochs=5000):
    for name, model, acc in [("MSE", model_mse, acc_mse), ("BCE", model_bce, acc_bce)]:
        assert acc is not None and len(acc) == n_epochs, (
            f"the {name} run must record {n_epochs} accuracies, one per epoch"
        )
        assert _fits_xor(model), f"the {name} network does not classify all four XOR points correctly"
    assert np.argmax(np.asarray(acc_bce) == 1.0) < np.argmax(np.asarray(acc_mse) == 1.0), (
        "the BCE network should fit XOR in fewer epochs than the MSE network: check the losses used"
    )
    print("MSE against BCE on XOR: all checks passed")


def xor_sgd_vs_adam(model_sgd, losses_sgd, model_adam, losses_adam, n_epochs=2000):
    for name, model, losses in [("SGD", model_sgd, losses_sgd), ("Adam", model_adam, losses_adam)]:
        assert losses is not None and len(losses) == n_epochs, (
            f"the {name} run must record {n_epochs} losses, one per epoch"
        )
        assert losses[-1] < losses[0], f"the loss of the {name} run did not decrease"
        assert _fits_xor(model), f"the {name} network does not classify all four XOR points correctly"
    assert model_sgd.layers[0].weight.data.shape == (4, 2), "the networks must have 4 hidden neurons"
    print("SGD against Adam on XOR: all checks passed")

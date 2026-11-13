"""Self-check for the function students write inside the Lab 7 notebook."""

import numpy as np


def check_early_stopping(EarlyStopping):
    class Array:                                       # all that EarlyStopping needs from a parameter or a buffer
        def __init__(self):
            self.data = np.zeros(3)

    class Model:                                       # ... and from a model
        def __init__(self):
            self.w, self.running = Array(), Array()

        def parameters(self):
            return [self.w]

        def buffers(self):
            return [self.running]

    def check(patience, losses, expected_stops, best_epoch, what):
        """Give `losses` to a fresh EarlyStopping, one per epoch. Before each call the parameter is set, in place,
        to the epoch number and the buffer to its opposite (a saved reference to an array would follow them)."""
        model, stopper, stops = Model(), EarlyStopping(patience), []
        for epoch, loss in enumerate(losses, start=1):
            model.w.data[...] = epoch
            model.running.data[...] = -epoch
            stop = stopper.step(loss, model)
            assert isinstance(stop, (bool, np.bool_)), "step must return True or False"
            stops.append(bool(stop))
        assert stops == expected_stops, \
            f"{what}: step returned {stops} for the losses {losses} with patience={patience}, expected {expected_stops}"
        model.w.data[...] = 99.0                       # the model has moved on since the best epoch
        model.running.data[...] = 99.0
        stopper.restore(model)
        assert np.all(model.w.data == best_epoch), \
            (f"{what}: restore must put back the parameters of epoch {best_epoch} "
             f"(got {model.w.data[0]}; are the saved values copies?)")
        assert np.all(model.running.data == -best_epoch), \
            (f"{what}: restore must also put back the buffers of epoch {best_epoch} "
             f"(got {model.running.data[0]}; are they saved and restored like the parameters?)")

    check(2, [1.0, 0.8, 0.9, 0.95], [False, False, False, True], 2,
          "stops after `patience` epochs without improvement")
    check(2, [1.0, 1.1, 0.9, 1.0, 1.1], [False, False, False, False, True], 3,
          "an improvement resets the counter")
    check(1, [1.0, 0.5, 0.6], [False, False, True], 2, "patience=1")
    check(3, [1.0, 0.9, 0.8, 0.7, 0.6], [False] * 5, 5, "no stop while the loss keeps improving")
    check(1, [1.0, 1.0], [False, True], 1, "an equal loss is not an improvement")
    print("early stopping: OK")

import numpy as np

from minitorch.transforms import RandomCrop


def test_randomcrop_shape_and_dtype():
    x = np.ones((7, 28, 28), dtype=np.uint8)
    out = RandomCrop(28, padding=2)(x)
    assert out.shape == (7, 28, 28) and out.dtype == x.dtype


def test_randomcrop_without_padding_is_identity():
    x = np.random.rand(5, 28, 28)
    assert np.array_equal(RandomCrop(28, padding=0)(x), x)


def test_randomcrop_smaller_window():
    assert RandomCrop(20, padding=0)(np.random.rand(4, 28, 28)).shape == (4, 20, 20)


def test_randomcrop_is_a_zero_padded_shift_with_independent_offsets():
    np.random.seed(0)
    img = np.random.rand(28, 28) + 1.0                 # no zeros, so offsets are identifiable
    padded = np.pad(img, 2)
    batch = np.repeat(img[None], 2000, axis=0)
    out = RandomCrop(28, padding=2)(batch)
    offsets = set()
    for crop in out:
        found = [(t, l) for t in range(5) for l in range(5)
                 if np.array_equal(crop, padded[t:t + 28, l:l + 28])]
        assert len(found) == 1                         # every output is one of the 25 shifts
        offsets.add(found[0])
    assert len(offsets) >= 20                          # offsets differ from image to image

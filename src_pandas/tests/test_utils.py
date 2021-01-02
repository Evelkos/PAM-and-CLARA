import numpy as np

from src.utils import compute_distance


def test_compute_distance():
    assert compute_distance(np.array([0, 0]), np.array([0, 9])) == 9

import numpy as np


def compute_distance(p1, p2):
    """
    Compute distance between two points.

    Arguments:
        p1: first point
        p2: second

    Return:
        Linear distance between two points.

    """
    return np.linalg.norm(p1 - p2)

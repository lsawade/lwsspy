import numpy as np


def norm1(d, s, w=1.0, n=1.0):
    return n * np.sum(w * np.abs(d-s))


def norm2(d, s, w=1.0, n=1.0):
    return 0.5 * n * np.sum(w * (d-s)**2)

import numpy as np


def norm1(d, w=1.0, n=1.0):
    return n * np.sum(w * np.abs(d))


def norm2(d, w=1.0, n=1.0):
    return 0.5 * n * np.sum(w * (d)**2)


def dnorm1(d, s, w=1.0, n=1.0):
    return n * np.sum(w * np.abs(d-s))


def dnorm2(d, s, w=1.0, n=1.0):
    return 0.5 * n * np.sum(w * (d-s)**2)

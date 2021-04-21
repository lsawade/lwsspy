import numpy as np


def lsq(self, d, s, w=1.0, n=1.0):
    return 0.5 * n * np.sum(w * (d-s)**2)

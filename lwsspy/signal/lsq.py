import numpy as np


def lsq(self, d, s):
    return 0.5 * np.sum((d-s)**2)

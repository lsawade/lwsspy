
import numpy as np


def dlna(d, s, w=1.0):
    return 0.5 * np.log(np.sum(w * d ** 2) / np.sum(w * s ** 2))

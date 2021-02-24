from scipy.signal import hilbert
import numpy as np


def envelope(array):
    return np.abs(hilbert(array))

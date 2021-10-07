import numpy as np
from typing import Iterable

def convm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Implements Matlab's conv(x, y, 'same')

    Args:
        x (np.ndarray):
            First Array
        y (np.ndarray):
            Second Array

    Returns:
        np.ndarray: Length(x) convolution of x with y.
    
    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)
    
    """
    npad = len(y) - 1
    full = np.convolve(x, y, 'full')
    first = npad - npad//2
    return full[first:first + len(x)]
    
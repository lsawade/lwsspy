import numpy as np


def magnitude(x: float):
    """Returns the magnitude of a number

    Args:
        x (float):
            just a float

    Returns"
        int:
            Magnitude of x

    """

    def magnitude(value):
        return np.floor(np.log10(np.abs(value))).astype(int)

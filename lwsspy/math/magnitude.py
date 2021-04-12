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
    if type(x) is list or type(x) is tuple:
        x = np.array(x)

    if type(x) != np.ndarray:
        if x != 0:
            return np.floor(np.log10(np.abs(x)))
        else:
            return 0
    else:
        out = np.zeros_like(x)
        out[x != 0] = np.floor(np.log10(np.abs(x[x != 0])))
        return out.astype(int)

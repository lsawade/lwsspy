import numpy as np


def cart2pol(x, y):
    """cartesian to polar coordinate

    Parameters
    ----------
    x : arraylike
        x coordinatees
    y : arraylike
        y coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

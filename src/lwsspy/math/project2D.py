import numpy as np
from scipy.interpolate import griddata


def project2D(x0: np.ndarray, y0: np.ndarray, xn: np.ndarray, yn: np.ndarray,
              x0q: np.ndarray, y0q: np.ndarray, **kwargs):
    """Takes in corresponding locations in 2 different coordinate systems
    and projects a set of query locations in the 0-coordinate system to
    the location in the n-coordinate system. Default griddata interpolation 
    method is 'linear'.

    Parameters
    ----------
    x0 : np.ndarray
        x-coordinate in the 0-coordinate system
    y0 : np.ndarray
        y-coordinate in the 0-coordinate system
    xn : np.ndarray
        x-coordinate in the n-coordinate system
    yn : np.ndarray
        y-coordinate in the n-coordinate system
    x0q : np.ndarray
        Queried x coordinate in the in 0 coordinate system to be
        transformed to n coordinate system
    y0q : np.ndarray
        Queried y coordinate in the in 0 coordinate system to be
        transformed to n coordinate system
    **kwargs :
        Sent to ``scipy.interpolate.griddata``

    Returns
    -------
    tuple
        xqn, yqn - the queried coordinates corresponding to the original

    Last modified: Lucas Sawade 2020.11.18 16:31 (lsawade@princeton.edu)
    """

    # Nice onliner-ish
    xqn = griddata(np.vstack((x0, y0)).T, xn, (x0q, y0q))
    yqn = griddata(np.vstack((x0, y0)).T, yn, (x0q, y0q))

    return xqn, yqn

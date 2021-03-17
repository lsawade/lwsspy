import numpy as np
from copy import deepcopy


def distance(x, y, x0: float, y0: float):
    """Compute euclidean 2D distance between points

    Parameters
    ----------
    x : arraylike
        x coordinates
    y : arraylike
        y coordinates
    x0 : float
        x coordinate to measure distance to
    y0 : float
        y coordinatte to measure distance to

    Returns
    -------
    float or arraylike
        distances

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.03.16 16.30
    """
    return np.sqrt((x - x0)**2 + (y - y0) ** 2)


def even2Dpoints(n, width, height, radius):
    """Creates evenly distributed points in 2D space.

    Parameters
    ----------
    n : int
        number of points to be generated
    width : floatt
        width of the 2D space
    height : float
        height of the 2D space
    radius : float
        min distance between the points

    Returns
    -------
    tuple
        x,y coordinates in 2D space
    """

    # Initialize empty array
    x = []
    y = []
    np.random.seed(1234)
    while len(x) < n:
        flag = True
        # Get candidate
        xc = (np.random.random(1)[0] - 0.5) * width
        yc = (np.random.random(1)[0] - 0.5) * height

        for ip in range(len(x)):
            if distance(x[ip], y[ip], xc, yc) < radius * 2:
                flag = False
                break

        if flag is True:
            x.append(deepcopy(xc))
            y.append(deepcopy(yc))

    return x, y

import numpy as np
from scipy import interpolate


def interpolatecurve(x, y, z, qz, **kwargs):
    """Not working as I though it would work..."""

    if "s" not in kwargs:
        s = 0
    else:
        s = kwargs['s']
        kwargs.pop('s')

    if "k" not in kwargs:
        k = 1
    else:
        k = kwargs['k']
        kwargs.pop('k')

    tck, u = interpolate.splprep([x, y, z], s=s, k=k, per=True, **kwargs)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    x_fine, y_fine, z_fine = interpolate.splev(qz, tck)
    return (x_fine, y_fine, z_fine), (x_knots, y_knots, z_knots)

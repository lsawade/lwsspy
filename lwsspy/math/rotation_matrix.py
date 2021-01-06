import numpy as np
import math


def rotation_matrix(axis, theta: float):
    """Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Axis is a vector

    Parameters
    ----------
    axis : list or tuple or numpy.ndarray
        3 component vector
    theta : float
        rotation angle

    Returns
    -------
    numpy.ndarray
        3x3 Matrix that rotates a 3 component vector in space.

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.06 11.00

    """

    # Convert possible list  to array
    axis = np.asarray(axis)

    # Do math
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    # Return rotation matrix
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

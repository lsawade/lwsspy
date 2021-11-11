import numpy as np
from ..math.geo2cart import geo2cart


def overlap(pointa1, pointa2, pointb1, pointb2):
    """Returns True if segments a and b overlap and False otherwise. ``pointa1``
    is the starting point of segment ``a`` and ``pointb2`` is the ending pointhr
    of segement ``b``. The points are given in geographical coordinates and the
    Earth is assumed to be spherical.

    Parameters
    ----------
    pointa1 : arraylike
        first segment starting point [lat, lon]
    pointa2 : arraylike
        first segment ending point [lat, lon]
    pointb1 : arraylike
        second segment starting point [lat, lon]
    pointb2 : arraylike
        second segment ending point [lat, lon]

    Returns
    -------
    [type]
        [description]
    """

    # Convert to cartesian
    a1 = geo2cart(1, *pointa1)
    a2 = geo2cart(1, *pointa2)
    b1 = geo2cart(1, *pointb1)
    b2 = geo2cart(1, *pointb2)

    # Find Normals of to the Great circles
    norma = np.cross(a1, a2)
    normb = np.cross(b1, b2)

    # Find the normals of the normals
    normab1 = np.cross(norma, normb)
    normab1 /= np.linalg.norm(normab1)
    normab2 = -normab1

    # Check distances
    ad = np.arccos(np.dot(a1, a2))
    ta1 = np.arccos(np.dot(normab1, a1))
    ta2 = np.arccos(np.dot(normab1, a2))
    bd = np.arccos(np.dot(b1, b2))
    tb1 = np.arccos(np.dot(normab1, b1))
    tb2 = np.arccos(np.dot(normab1, b2))

    ra1 = np.arccos(np.dot(normab2, a1))
    ra2 = np.arccos(np.dot(normab2, a2))
    rb1 = np.arccos(np.dot(normab2, b1))
    rb2 = np.arccos(np.dot(normab2, b2))

    return (np.isclose(ad, ta1+ta2) and np.isclose(bd, tb1+tb2)) or \
        (np.isclose(ad, ra1+ra2) and np.isclose(bd, rb1+rb2))

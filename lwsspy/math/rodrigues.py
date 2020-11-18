from numpy import sin, cos, dot, eye
from numpy.linalg import norm
import numpy as np


def rodrigues(r):
    """Matrix that rotates points so that points have new orientation.

    Parameters
    ----------
    r : numpy.ndarray
        new normal vector

    Edited from:
    https://github.com/robEllenberg/comps-plugins/blob/master/python/rodrigues.py

    Theory:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Last modified: Lucas Sawade, 2020.11.16 16.30.00 (lsawade@princeton.edu)

    """
    def S(n):
        Sn = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        return Sn
    theta = norm(r)
    if theta > 1e-30:
        n = r/theta
        Sn = S(n)
        R = eye(3) + sin(theta)*Sn + (1-cos(theta))*dot(Sn, Sn)
    else:
        Sr = S(r)
        theta2 = theta**2
        R = eye(3) + (1-theta2/6.)*Sr + (.5-theta2/24.)*dot(Sr, Sr)
    return R

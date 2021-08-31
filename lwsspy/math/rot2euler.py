"""
Taken from pyrocko on July 27 2021, 16:30
https://pyrocko.org/docs/current/_modules/pyrocko/moment_tensor.html#MomentTensor.both_strike_dip_rake

"""

import numpy as np
import math


def matrix_to_euler(rotmat):
    '''Get eulerian angle triplet from rotation matrix.'''

    ex = cvec(1., 0., 0.)
    ez = cvec(0., 0., 1.)
    exs = rotmat.T * ex
    ezs = rotmat.T * ez
    enodes = np.cross(ez.T, ezs.T).T
    if np.linalg.norm(enodes) < 1e-10:
        enodes = exs
    enodess = rotmat*enodes
    cos_alpha = float((ez.T*ezs))
    if cos_alpha > 1.:
        cos_alpha = 1.

    if cos_alpha < -1.:
        cos_alpha = -1.

    alpha = math.acos(cos_alpha)
    beta = np.mod(math.atan2(enodes[1, 0], enodes[0, 0]), math.pi*2.)
    gamma = np.mod(-math.atan2(enodess[1, 0], enodess[0, 0]), math.pi*2.)

    return unique_euler(alpha, beta, gamma)


def unique_euler(alpha, beta, gamma):
    '''Uniquify eulerian angle triplet.

    Put eulerian angle triplet into ranges compatible with
    ``(dip, strike, -rake)`` conventions in seismology::

        alpha (dip)   : [0, pi/2]
        beta (strike) : [0, 2*pi)
        gamma (-rake) : [-pi, pi)

    If ``alpha1`` is near to zero, ``beta`` is replaced by ``beta+gamma`` and
    ``gamma`` is set to zero, to prevent this additional ambiguity.

    If ``alpha`` is near to ``pi/2``, ``beta`` is put into the range
    ``[0,pi)``.
    '''

    pi = math.pi

    alpha = np.mod(alpha, 2.0*pi)

    if 0.5*pi < alpha and alpha <= pi:
        alpha = pi - alpha
        beta = beta + pi
        gamma = 2.0*pi - gamma
    elif pi < alpha and alpha <= 1.5*pi:
        alpha = alpha - pi
        gamma = pi - gamma
    elif 1.5*pi < alpha and alpha <= 2.0*pi:
        alpha = 2.0*pi - alpha
        beta = beta + pi
        gamma = pi + gamma

    alpha = np.mod(alpha, 2.0*pi)
    beta = np.mod(beta,  2.0*pi)
    gamma = np.mod(gamma+pi, 2.0*pi)-pi

    # If dip is exactly 90 degrees, one is still
    # free to choose between looking at the plane from either side.
    # Choose to look at such that beta is in the range [0,180)

    # This should prevent some problems, when dip is close to 90 degrees:
    if abs(alpha - 0.5*pi) < 1e-10:
        alpha = 0.5*pi

    if abs(beta - pi) < 1e-10:
        beta = pi

    if abs(beta - 2.*pi) < 1e-10:
        beta = 0.

    if abs(beta) < 1e-10:
        beta = 0.

    if alpha == 0.5*pi and beta >= pi:
        gamma = - gamma
        beta = np.mod(beta-pi,  2.0*pi)
        gamma = np.mod(gamma+pi, 2.0*pi)-pi
        assert 0. <= beta < pi
        assert -pi <= gamma < pi

    if alpha < 1e-7:
        beta = np.mod(beta + gamma, 2.0*pi)
        gamma = 0.

    return (alpha, beta, gamma)


def cvec(x, y, z):
    return np.matrix([[x, y, z]], dtype=np.float).T


def rvec(x, y, z):
    return np.matrix([[x, y, z]], dtype=np.float)


def eigh_check(a):
    evals, evecs = np.linalg.eigh(a)
    assert evals[0] <= evals[1] <= evals[2]
    return evals, evecs

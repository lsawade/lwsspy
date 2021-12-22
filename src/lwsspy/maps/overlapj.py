from jax import jit
import jax.numpy as jnp
from jax.lax import cond
from numpy import logical_and
from ..math.geo2cartj import geo2cartj


def compute_overlap(pointa1, pointa2, pointb1, pointb2):

    # Convert to cartesian
    a1 = geo2cartj(1, *pointa1)
    a2 = geo2cartj(1, *pointa2)
    b1 = geo2cartj(1, *pointb1)
    b2 = geo2cartj(1, *pointb2)

    # Find Normals of to the Great circles
    norma = jnp.cross(a1, a2)
    normb = jnp.cross(b1, b2)

    # Find the normals of the normals
    normab1 = jnp.cross(norma, normb)
    normab1 /= jnp.linalg.norm(normab1)
    normab2 = -normab1

    # Check distances
    ad = jnp.arccos(jnp.dot(a1, a2))
    ta1 = jnp.arccos(jnp.dot(normab1, a1))
    ta2 = jnp.arccos(jnp.dot(normab1, a2))
    bd = jnp.arccos(jnp.dot(b1, b2))
    tb1 = jnp.arccos(jnp.dot(normab1, b1))
    tb2 = jnp.arccos(jnp.dot(normab1, b2))

    ra1 = jnp.arccos(jnp.dot(normab2, a1))
    ra2 = jnp.arccos(jnp.dot(normab2, a2))
    rb1 = jnp.arccos(jnp.dot(normab2, b1))
    rb2 = jnp.arccos(jnp.dot(normab2, b2))

    check1 = jnp.logical_and(
        jnp.allclose(ad, ta1+ta2), jnp.allclose(bd, tb1+tb2))
    check2 = jnp.logical_and(
        jnp.allclose(ad, ra1+ra2), jnp.allclose(bd, rb1+rb2))

    return jnp.logical_or(check1, check2)


def truefun(check1, check2, pointa1,
            pointa2, pointb1, pointb2):
    return check1


def overlapj(pointa1, pointa2, pointb1, pointb2):
    """Returns True if segments a and b overlap and False otherwise. ``pointa1``
    is the starting point of segment ``a`` and ``pointb2`` is the ending point
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

    # Check if segments are the same
    check1 = jnp.logical_or(
        jnp.logical_and(jnp.allclose(pointa1, pointb1),
                        jnp.allclose(pointa2, pointb2)),
        jnp.logical_and(jnp.allclose(pointa1, pointb2),
                        jnp.allclose(pointa2, pointb1)))

    # If segments are not the same, check if the have the same endpoints
    check2 = jnp.logical_or(
        jnp.logical_or(
            jnp.allclose(pointa1, pointb1),
            jnp.allclose(pointa1, pointb2)),
        jnp.logical_or(
            jnp.allclose(pointa2, pointb2),
            jnp.allclose(pointa2, pointb1)),
    )

    print(jnp.logical_or(check1, check2))
    return check1 if jnp.logical_or(check1, check2) else compute_overlap(pointa1, pointa2, pointb1, pointb2)

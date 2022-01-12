import jax.numpy as jnp
from jax import jit


@jit
def geo2cartj(r: float or jnp.ndarray or list,
              theta: float or jnp.ndarray or list,
              phi: float or jnp.ndarray or list):
    """Computes Cartesian coordinates from geographical coordinates.

    Parameters
    ----------
    r : float or numpy.ndarray or list
        Radius
    theta : float or numpy.ndarray or list
        Latitude (-90, 90)
    phi : float or numpy.ndarray or list
        Longitude (-180, 180)

    Returns
    -------
    float or jnp.ndarray or list, float or jnp.ndarray or list, float or jnp.ndarray or list
        (x, y, z)
    """

    if type(r) is list:
        r = jnp.array(r)
        theta = jnp.array(theta)
        phi = jnp.array(phi)

    # Convert to Radians
    thetarad = theta * jnp.pi/180.0
    phirad = phi * jnp.pi/180.0

    # Compute Transformation
    x = r * jnp.cos(thetarad) * jnp.cos(phirad)
    y = r * jnp.cos(thetarad) * jnp.sin(phirad)
    z = r * jnp.sin(thetarad)

    if type(r) is list:
        x = x.tolist()
        y = y.tolist()
        z = z.tolist()

    return jnp.array([x, y, z])

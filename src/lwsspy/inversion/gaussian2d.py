import os
# import jax.numpy as jnp
import numpy as np


def gaussian2d(
        params,
        x):
    """Generates arbitrary gaussian distribution for the corresponding arrays
    x and, at expected value (xo, yo), and standard deviations
    (sigma_x, sigma_y). To ensure a possibilty of angled distributions
    and offset distributions model parameters theta and offset are included.

    Args:
        x (np.ndarray):
            x coordinates
        y (np.ndarray):
            y coordinates
        amplitude (float):
            amplitude of the Gaussian function
        xo (float):
            center in x direction
        yo (float):
            center in y direction
        sigma_x ():
            std in x direction
        sigma_y (float):
            std in y direction
        theta (float):
            angle of the distribution
        offset (float):
            offset of the distribution (could be gaussian + offset)

    Returns:
        G(x, y) as np.ndarray

    Last modified: Lucas Sawade, 2020.09.15 19.44 (lsawade@princeton.edu)
    """
    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = params
    x, y = x

    a = (jnp.cos(theta)**2)/(2*sigma_x**2) + (jnp.sin(theta)**2)/(2*sigma_y**2)
    b = -(jnp.sin(2*theta))/(4*sigma_x**2) + (jnp.sin(2*theta))/(4*sigma_y**2)
    c = (jnp.sin(theta)**2)/(2*sigma_x**2) + (jnp.cos(theta)**2)/(2*sigma_y**2)

    g = offset + amplitude * \
        jnp.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

    return g

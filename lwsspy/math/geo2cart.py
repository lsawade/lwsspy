import numpy as np


def geo2cart(r: float or np.ndarray or list,
             theta: float or np.ndarray or list,
             phi: float or np.ndarray or list) \
    -> (float or np.ndarray or list,
        float or np.ndarray or list,
        float or np.ndarray or list):
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
    float or np.ndarray or list, float or np.ndarray or list, float or np.ndarray or list
        (x, y, z)
    """

    if type(r) is list:
        r = np.array(r)
        theta = np.array(theta)
        phi = np.array(phi)

    # Convert to Radians
    thetarad = theta * np.pi/180.0
    phirad = phi * np.pi/180.0

    # Compute Transformation
    x = r * np.cos(thetarad) * np.cos(phirad)
    y = r * np.cos(thetarad) * np.sin(phirad)
    z = r * np.sin(thetarad)

    if type(r) is list:
        x = x.tolist()
        y = y.tolist()
        z = z.tolist()

    return x, y, z

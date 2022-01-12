
from cartopy.crs import Projection


def threshproj(proj: Projection, threshholdfactor: float = 10):
    """Lowers plotting threshold for projection. Great for circles.

    Parameters
    ----------
    proj : Projection
        cartopy projection
    threshfactor : float, optional
        divides original threshold by this value, by default 10
    """
    proj._threshold = proj._threshold/threshholdfactor

    return proj

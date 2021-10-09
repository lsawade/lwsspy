import numpy as np
from scipy.spatial import KDTree


def compute_dist2D(x0, y0, x1, y1):
    """Convenience Function for 2D space. Get indeces of points ``1`` that are 
    closest to points ``0``. Note that ``0`` could be a single point.

    Takes two location vectors computes there distance from each element
    in the first location to the elements in the location.

    This is done using a kdtree algorithm."""

    # Create point arrays
    xy0 = np.array([[*x0], [*y0]]).T
    xy1 = np.array([[*x1], [*y1]]).T

    # Create kdtree
    mytree = KDTree(xy0)
    dist, indexes = mytree.query(xy1)

    return dist, indexes

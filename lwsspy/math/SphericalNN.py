"""
Taken from ObsPy originally, but heavily modified.

"""
# External
from typing import List
import numpy as np
from scipy.spatial import cKDTree
from obspy.core.inventory import Station

# Internal
import lwsspy as lpy


class SphericalNN(object):
    """
    Spherical nearest neighbour queries using scipy's fast kd-tree
    implementation.
    """

    def __init__(self, lat, lon):
        cart_data = self.spherical2cartesian(lat, lon)
        self.data = cart_data
        self.kd_tree = cKDTree(data=cart_data, leafsize=10)

    def query(self, qlat, qlon):
        points = self.spherical2cartesian(qlat, qlon)
        d, i = self.kd_tree.query(points)

        # Filter NaNs. Happens when not enough points are available.
        m = np.isfinite(d)
        return d[m], i[m]

    def query_pairs(self, maximum_distance):
        return self.kd_tree.query_pairs(maximum_distance)

    def interp(self, data, qlat, qlon, maximum_distance=None, no_weighting=False):
        """spherical interpolation function using the ``SphericalNN`` object.

        Parameters
        ----------
        data : numpy.ndarray
            data
        qlat : numpy.ndarray
            query latitudes
        qlon : numpy.ndarray
            query longitude
        maximum_distance : float, optional
            max distace for the interpolation in degree angle. Default Infinity.
            If the mindistance to any points is larger than maximum_distance the
            interpolated value is set to ``np.nan``.
        no_weighting : bool, optional
            Whether or not the function uses a weightied nearest neighbor
            interpolation
        """

        # Get query points in cartesian
        shp = qlat.shape
        points = self.spherical2cartesian(qlat.flatten(), qlon.flatten())

        # Query points
        if no_weighting:
            # Get distances and indeces
            d, inds = self.kd_tree.query(points)

            # Assign the interpolation data.
            qdata = data[inds]

        else:

            # Get multiple distances and indeces
            d, inds = self.kd_tree.query(points, k=10)

            # Actual weighted interpolation.
            w = (1-d / np.max(d, axis=1)[:, np.newaxis]) ** 2

            # Double check distance
            mind = np.min(d, axis=1)

            # Take things that are further than a certain distance
            qdata = np.sum(w * data[inds], axis=1) / np.sum(w, axis=1)

        # Filter out distances too far out.
        if maximum_distance is not None:
            qdata = np.where(
                mind <= 2 * np.sin(maximum_distance/2.0/180.0*np.pi)
                * lpy.EARTH_RADIUS_KM,
                qdata, np.nan)

        return qdata.reshape(shp)

    @staticmethod
    def spherical2cartesian(lat, lon):
        """
        Converts a list of :class:`~obspy.fdsn.download_status.Station`
        objects to an array of shape(len(list), 3) containing x/y/z in meters.
        """
        # Create three arrays containing lat/lng/radius.
        r = np.ones_like(lat) * lpy.EARTH_RADIUS_KM

        # Convert data from lat/lng to x/y/z.
        x, y, z = lpy.geo2cart(r, lat, lon)

        return np.vstack((x, y, z)).T

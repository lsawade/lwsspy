"""
Taken from ObsPy originally, but heavily modified.

"""
# External
from __future__ import annotations
from typing import Optional
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree

# Internal
from .. import base as lbase
from .. import math as lmat


class SphericalNN3D(object):
    """Spherical nearest neighbour queries using scipy's fast kd-tree
    implementation.

    Attributes
    ----------
    data : numpy.ndarray
        cartesian point data array [x,y,z]
    kd_tree : scipy.spatial.cKDTree
        a KDTree used to query data


    Methods
    -------
    query(qlat, qlon)
        Query a set of latitudes and longitudes
    query_pairs(maximum_distance)
        Find pairs of points that are within a certain distance of each other
    interp(data, qlat, qlon)
        Use the kdtree to interpolate data corresponding
        to the points of the Kdtree onto a new set of points using nearest
        neighbor interpolation or weighted nearest neighbor
        interpolation (default).

    Notes
    -----

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.07 19.00

    """

    def __init__(self, lat, lon, d):
        """Initialize class

        Parameters
        ----------
        lat : numpy.ndarray
            latitudes
        lon : numpy.ndarray
            longitudes
        """
        cart_data = self.spherical2cartesian(lat, lon, d)
        self.data = cart_data
        self.kd_tree = cKDTree(data=cart_data, leafsize=10)

    def query(self, qlat, qlon, qd=0.0, maximum_distance=np.inf, **kwargs):
        """Query latitude and longitude values from kdtree

        Parameters
        ----------
        qlat : np.ndarray or float
            query latitude
        qlon : np.ndarray or float
            query longitude
        qdep : np.ndarray or float
            query longitude

        Returns
        -------
        tuple
            (distance, indeces) to closest point in kdtree
        """
        distkm = np.abs(2*np.sin(maximum_distance/2.0 /
                                 180.0*np.pi))*lbase.EARTH_RADIUS_KM
        points = self.spherical2cartesian(qlat, qlon, d=qd)
        d, i = self.kd_tree.query(
            points, distance_upper_bound=distkm, **kwargs)

        # Filter NaNs. Happens when not enough points are available.
        m = np.isfinite(d)
        pos = np.where(m)[0]

        return d[m], i[m], pos

    # def query_pairs(self, maximum_distance=180.0):
    #     """Query pairs within the kdtree

    #     Parameters
    #     ----------
    #     maximum_distance : float
    #         Maximum query distance in deg

    #     Returns
    #     -------
    #     set or ndarray
    #         Set of pairs (i, j) where i < j
    #     """
    #     distkm = np.abs(2 * np.sin(maximum_distance/2.0 /
    #                                180.0*np.pi)) * lbase.EARTH_RADIUS_KM
    #     return self.kd_tree.query_pairs(distkm, output_type='ndarray')

    # def sparse_distance_matrix(self, other: Union[SphericalNN3D, None] = None,
    #                            maximum_distance=180.0, sparse: bool = False,
    #                            km: bool = False):
    #     """Computes the sparse distance matrix between two kdtree. if no other
    #     kdtree is provided, this kdtree is used

    #     Parameters
    #     ----------
    #     other : SphericalNN
    #         other SphericalNN tree
    #     maximum_distance : float
    #         Maximum query distance in deg
    #     sparse: bool
    #         Flag to output sparse array for memory. Default False
    #     km: bool
    #         flag to output distances in kilometers. Default False

    #     Returns
    #     -------
    #     set or ndarray
    #         Set of pairs (i, j) where i < j
    #     """
    #     # Get distance
    #     distkm = np.abs(2 * np.sin(maximum_distance/2.0 /
    #                                180.0*np.pi)) * lbase.EARTH_RADIUS_KM

    #     # Get tree
    #     if isinstance(other, SphericalNN3D):
    #         other = other.kd_tree
    #     else:
    #         other = deepcopy(self).kd_tree

    #     # Compute dense or sparse distance matrix
    #     if sparse:
    #         output_mat = self.kd_tree.sparse_distance_matrix(
    #             other, distkm)
    #     else:
    #         output_mat = self.kd_tree.sparse_distance_matrix(
    #             other, distkm).toarray()

    #     # Convert form kilometers to degrees.
    #     if km is False:
    #         output_mat *= lbase.KM2DEG

    #     return output_mat

    # def interpolator(self, qlat, qlon, maximum_distance=None,
    #                  no_weighting=False, k: Optional[int] = None, p: float = 2.0):
    #     """Spherical interpolation function using the ``SphericalNN`` object.
    #     Returns an interpolator that can be used for interpolating the same
    #     set of locations based on the KDTree. The only input the interpolator
    #     takes are the data corresponding to the points in the KDTree.

    #     Parameters
    #     ----------
    #     qlat : numpy.ndarray
    #         query latitudes
    #     qlon : numpy.ndarray
    #         query longitude
    #     maximum_distance : float, optional
    #         max distace for the interpolation in degree angle. Default None.
    #         If the mindistance to any points is larger than maximum_distance the
    #         interpolated value is set to ``np.nan``.
    #     no_weighting : bool, optional
    #         Whether or not the function uses a weightied nearest neighbor
    #         interpolation
    #     k : int, optional
    #         Define maximum number of neighbors to be used for the weighted
    #         interpolation. Not used if ``no_weighting = True``. Default None
    #     p : float, optional
    #         Exponent to compute the inverse distance weights. Note that in
    #         the limit ``p->inf`` is just a nearest neighbor interpolation.
    #         Default is 2

    #     Notes
    #     -----

    #     In the future, I may add a variable weighting function for the
    #     weighted interpolation.

    #     Please refer to https://en.wikipedia.org/wiki/Inverse_distance_weighting
    #     for the interpolation weighting.

    #     """

    #     # Get query points in cartesian
    #     shp = qlat.shape
    #     points = self.spherical2cartesian(qlat.flatten(), qlon.flatten())

    #     # Query points
    #     if no_weighting:
    #         # Get distances and indeces
    #         d, inds = self.kd_tree.query(points)

    #         def interpolator(data):

    #             # Assign the interpolation data.
    #             qdata = data[inds]

    #             # Filter out distances too far out.
    #             if maximum_distance is not None:
    #                 qdata = np.where(
    #                     d <= np.abs(
    #                         2 * np.sin(maximum_distance/2.0/180.0*np.pi))
    #                     * lbase.EARTH_RADIUS_KM,
    #                     qdata, np.nan)

    #             return data[inds].reshape(shp)

    #     else:

    #         # Set K to the max number of points if not given
    #         if k is None:
    #             k = self.kd_tree.n
    #         else:
    #             if k > self.kd_tree.n:
    #                 k = self.kd_tree.n

    #         # Get multiple distances and indeces
    #         d, inds = self.kd_tree.query(points, k=k)

    #         if d.shape == (1,):
    #             d = d.reshape((1, 1))
    #             inds = inds.reshape((1, 1))

    #         # Filter out distances too far out.
    #         # Modified Shepard's method
    #         if maximum_distance is not None:
    #             # Get cartesian distance
    #             cartd = np.abs(2 * np.sin(maximum_distance/2.0/180.0*np.pi)
    #                            * lbase.EARTH_RADIUS_KM)

    #             # Compute the weights using modified shepard
    #             w = (np.fmax(0, cartd - d) / cartd * d) ** 2
    #             wsum = np.sum(w, axis=1)
    #             datarows = ~np.isclose(wsum, 0.0)

    #             def interpolator(data):
    #                 """Using the weights and indices, we can return an interpolator
    #                 """

    #                 # Empty array
    #                 qdata = np.empty_like(wsum, dtype=np.float32)
    #                 qdata[:] = np.nan

    #                 # Check which rows only contain nans
    #                 tmp = w[datarows, :] * data[inds[datarows]]
    #                 nanrows = np.isnan(tmp[:, 0])

    #                 # Do interpolation and replace only-nan rows with np.nan
    #                 tmp2 = np.nansum(tmp, axis=1) / wsum[datarows]
    #                 tmp2[nanrows] = np.nan

    #                 qdata[datarows] = tmp2

    #                 return qdata.reshape(shp)

    #         # Shepard's method
    #         else:

    #             # Compute the weights using modified shepard
    #             w = (1 / d) ** p
    #             wsum = np.sum(w, axis=1)
    #             datarows = ~np.isclose(wsum, 0.0)

    #             def interpolator(data):
    #                 """Using the weights and indices, we can return an interpolator
    #                 """

    #                 # Empty array
    #                 qdata = np.empty_like(wsum, dtype=np.float32)
    #                 qdata[:] = np.nan

    #                 # Check which rows only contain nans
    #                 tmp = w * data[inds[datarows]]
    #                 nanrows = np.isnan(tmp[:, 0])

    #                 # interpolation using the weights
    #                 qdata[datarows] = np.sum(
    #                     tmp, axis=1) / wsum[datarows]

    #                 qdata[datarows][nanrows] = np.nan

    #                 return qdata.reshape(shp)

    #     return interpolator

    @ staticmethod
    def spherical2cartesian(lat, lon, d=0.0):
        """
        Converts a list of :class:`~obspy.fdsn.download_status.Station`
        objects to an array of shape(len(list), 3) containing x/y/z in meters.
        """
        # Create three arrays containing lat/lng/radius.
        r = np.ones_like(lat) * lbase.EARTH_RADIUS_KM

        # Convert data from lat/lng to x/y/z.
        x, y, z = lmat.geo2cart(r - d, lat, lon)

        return np.vstack((x, y, z)).T

from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d
from .reckon import reckon
from cartopy.geodesic import Geodesic


def gctrack(lat, lon, dist: float = 1.0) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """Given waypoints and a point distance, this function computes evenly
    spaced points along the great circle and the given waypoints.

    Parameters
    ----------
    lat : np.ndarray
        waypoint latitudes
    lon : np.ndarray
        waypoint longitudes
    dist : float
        distance in degrees


    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray)
        latitudes, longitudes, distances


    Notes
    -----

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.09.15 21.11

    """

    # First get distances between points
    N = len(lon)
    dists = np.zeros(N-1)
    az = np.zeros(N-1)

    # Cumulative
    cdists = np.zeros(N)

    # Create Geodesic class
    G = Geodesic()

    # Get distances along the waypoints
    mat = np.asarray(G.inverse(np.array((lon[0:-1], lat[0:-1])).T,
                               np.array((lon[1:], lat[1:])).T))
    dists = mat[:, 0]/1000.0/111.11
    az = mat[:, 1]

    # Get tracks between segments that are far apart
    tracks = []
    for _i in range(N-1):

        if dists[_i] > dist:
            # Create vector between two poitns
            trackdists = np.arange(0, dists[_i], dist)
            track = np.array(reckon(lat[_i], lon[_i], trackdists, az[_i]))

        else:
            track = np.array((lat[_i:_i+1], lon[_i:_i+1]))

        tracks.append(track)

    # Add last point because usually not added
    tracks.append(np.array(((lat[-1], lon[-1]),)).T)

    # Get tracks
    utrack = np.hstack(tracks).T

    # Remove duplicates if there are any
    _, idx = np.unique(utrack, return_index=True, axis=0)
    utrack = utrack[np.sort(idx), :]

    # Get distances along the new track
    mat = np.asarray(G.inverse(
        np.array((utrack[0:-1, 1], utrack[0:-1, 0])).T,
        np.array((utrack[1:, 1],   utrack[1:, 0])).T))
    udists = mat[:, 0]/1000.0/111.11

    # Compute cumulative distance
    M = len(utrack[:, 0])
    cdists = np.zeros(M)
    cdists[1:] = np.cumsum(udists)

    # Interpolate to the final vectors
    maxdist = np.max(cdists)
    qdists = np.linspace(0, maxdist, int(maxdist/dist))
    ilat = interp1d(cdists, utrack[:, 0])
    ilon = interp1d(cdists, utrack[:, 1])
    qlat, qlon = ilat(qdists), ilon(qdists)

    return qlat, qlon, qdists

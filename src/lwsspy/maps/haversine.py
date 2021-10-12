import numpy as np


def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    Taken from here:
    https://stackoverflow.com/a/29546836/13239311



    Parameters
    ----------
    lon1 : array
        longitude 1
    lat1 : array
        latitude 1
    lon2 : array
        longitude 2
    lat2 : array
        latitude 2

    Returns
    -------
    array
        distance in km for a spherical earth with r = 6371 km.
    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

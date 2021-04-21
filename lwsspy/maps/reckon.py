import numpy as np


def reckon(lat, lon, distance, bearing):
    """ Computes new latitude and longitude from bearing and distance.

    Parameters
    ----------
    lat: in degrees
    lon: in degrees
    bearing: in degrees
    distance: in degrees

    Returns
    -------
    lat, lon


    lat1 = math.radians(52.20472)  # Current lat point converted to radians
    lon1 = math.radians(0.14056)  # Current long point converted to radians
    bearing = np.pi/2 # 90 degrees
    # lat2  52.20444 - the lat result I'm hoping for
    # lon2  0.36056 - the long result I'm hoping for.

    """

    # Convert degrees to radians for numpy
    lat1 = lat/180*np.pi
    lon1 = lon/180 * np.pi
    brng = bearing/180*np.pi
    d = distance/180*np.pi

    # Compute latitude
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d)
                     + np.cos(lat1) * np.sin(d) * np.cos(brng))

    # Compute longitude
    lon2 = lon1 + np.arctan2(np.sin(brng) * np.sin(d) * np.cos(lat1),
                             np.cos(d) - np.sin(lat1) * np.sin(lat2))

    # Convert back
    lat2 = lat2/np.pi*180
    lon2 = lon2/np.pi*180

    return lat2, lon2

import numpy as np
from ..math.geo2cart import geo2cart
from ..math.cart2geo import cart2geo


def geomidpoint(lat1, lon1, lat2, lon2):
    deg2rad = np.pi/180.0
    rad2deg = 180.0/np.pi

    # Convert to radians
    lat1 *= deg2rad
    lat2 *= deg2rad
    lon1 *= deg2rad
    lon2 *= deg2rad

    # Subcalcs
    dlon = lon2-lon1
    Bx = np.cos(lat2) * np.cos(dlon)
    By = np.cos(lat2) * np.sin(dlon)

    # Main calcs
    lat3 = np.arctan2(
        np.sin(lat1)+np.sin(lat2),
        np.sqrt( (np.cos(lat1)+Bx)*(np.cos(lat1)+Bx) + By*By ) );
    lon3 = lon1 + np.arctan2(By, np.cos(lat1) + Bx)

    # Convert back to degress
    lat3 *= rad2deg
    lon3 *= rad2deg

    return lat3, lon3


def geomidpointv(lat1, lon1, lat2, lon2):
    """This is dumb simple..."""
    
    # Gets cartesian coordinates
    x1 = geo2cart(1, lat1, lon1)
    x2 = geo2cart(1, lat2, lon2)

    # vector between x1 and x2
    dx = np.array(x2) - np.array(x1)

    # Add vector to point
    _, lat3, lon3 = cart2geo(*(x1 + 0.5 * dx))

    return lat3, lon3


import typing as tp
import numpy as np
from torch import deg2rad, rad2deg
from .haversine import haversine
from ..math.geo2cart import geo2cart
from ..math.cart2geo import cart2geo


def geofracpoint(
    lat1, lon1, lat2, lon2,
    fac: tp.Union[tp.Iterable[float], float] = 0.5):
    
    """Create squence of points along Great circle span by to geolocations.
    Uses the relationshipt between the fraction of the angle on the great circle
    and distance alonf the ruhmb line to compute Geographic point on the
    surface.

    Parameters
    ----------
    lat1 : float
        latitude of point 1
    lon1 : float
        longitude of point 1
    lat2 : float
        latitude of point 2
    lon2 : float
        longitude of point 2
    fac : tp.Union[tp.Iterable[float], float], optional
        Could be a single value or an iterable. Used to compute the
        fractional epicentral distance between the points, by default 0.5

    Returns
    -------
    tuple
        either single point (lat, lon)
        or to lists (List[lat], List[lon])

    """

    # See haversine!
    RE = 6371.0
    rad2deg = 180.0/np.pi

    # Gets cartesian coordinates
    x1 = geo2cart(RE, lat1, lon1)
    x2 = geo2cart(RE, lat2, lon2)

    # vector between x1 and x2
    dx = np.array(x2) - np.array(x1)
    normdx = np.sqrt(np.sum(dx**2))

    # Get epicentral distance
    deltarad = haversine(lat1, lon1, lat2, lon2)/RE

    # halfcord
    hchord = RE*np.sin(deltarad/2.0) # also norm(dx)/2

    # Origin to chord
    Ochord = RE*np.cos(deltarad/2.0)

    # Fix whether a single or multiple places are given
    floatfac = False

    if isinstance(fac, float) or isinstance(fac, int):
        fac = [fac]
        floatfac = True

    # Empty list
    lat3 = []
    lon3 = []

    for _f in fac:
    
        # Get fractional angle
        fdeltarad = _f * deltarad

        # Get dx allong chord from Intersection ot halfchord and ochord to
        # fractional angle
        if _f == 0.5:
            chordfrac = 0.5

        elif _f < 0.5:
            # Get angle from Ochrod line to fractional angle line
            counterrad = deltarad/2 - fdeltarad
            counterdx = np.tan(counterrad) * Ochord

            # Get dx fraction along chord
            chordfrac = (normdx/2-counterdx)/normdx

        else:
            extrarad = fdeltarad - deltarad/2
            print(extrarad*rad2deg)
            extradx = Ochord*np.tan(extrarad)

            # Get dx fraction along chord
            chordfrac = (hchord+extradx)/normdx

        # Add vector to point
        _, tlat3, tlon3 = cart2geo(*(x1 + chordfrac * dx))

        # Add to list 
        lat3.append(tlat3)
        lon3.append(tlon3)

    if floatfac:
        return lat3[0], lon3[0]
    else:
        return lat3, lon3

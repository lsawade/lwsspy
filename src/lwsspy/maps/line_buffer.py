import numpy as np
from ..math.geo2cart import geo2cart
from ..math.cart2geo import cart2geo
from ..math.Ra2b import Ra2b
from shapely.geometry import Polygon
from shapely.ops import cascaded_union


def line_buffer(lat, lon, delta=1.0, c180=False):
    """Creates buffer around geographical line using from pole rotated cap
    outlines. This is far from perfect, but it does "a" job. There isn't much
    more that one can and be spherically accurate. As long as the amount of
    points in the line is fair this should have no problem to immitate a true
    buffer.

    Parameters
    ----------
    lat : np.ndarray
        latitudes
    lon : np.ndarray
        longitudes
    delta : float, optional
        epicentral distance to line, by default 1.0

    Returns
    -------
    np.ndarray
        Nx2 matrix with coordinates of a polygon that forms a buffer around
        the line

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.04.17 00.30

    """

    # Buffer circle around z axis
    r = 1
    phi = np.arange(0, 2*np.pi, 2*np.pi/100)
    theta = delta*np.pi/180

    # Circle around z axis in cartesian
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)*np.ones_like(phi)
    points = np.vstack((x, y, z))

    # Get circles around the local coordinates
    rcoordset = []

    for _lat, _lon in zip(lat, lon):

        # Convert line coordinates
        x1 = geo2cart(1, _lat, _lon)

        # Compute rotation matrix compute the new set of points
        rpoints = Ra2b((0, 0, 1), x1) @ points

        # Append x and y coordinates
        rcoordset.append(cart2geo(
            rpoints[0, :], rpoints[1, :], rpoints[2, :]))

    # Generate and combine polygons
    polygons = []
    circles = []

    for _i, coords in enumerate(rcoordset):
        lat = coords[1]
        lon = coords[2]

        if c180 is True:
            lon = np.where(lon < 0.0, lon + 360.0, lon)

        circles.append((lon, lat))
        polygon = Polygon([(_x, _y)
                           for _x, _y in zip(lon, lat)])
        polygons.append(polygon)

    # Combine from converted points
    upoly = cascaded_union(polygons)

    if upoly.type == 'MultiPolygon':
        polyplot = [np.array(x.exterior.xy).T for x in upoly.geoms]
    else:
        polyplot = [np.array(upoly.exterior.xy).T]

    return polyplot, circles

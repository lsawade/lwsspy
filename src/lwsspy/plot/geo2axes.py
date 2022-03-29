from cartopy.crs import Geodetic
from cartopy.mpl.geoaxes import GeoAxes


def geo2axes(ax: GeoAxes, lon: float, lat: float):
    """Converts a geographical point in terms of lat and lon to fractinoal axes 
    coordinates (think fractional points of a bbox)

    Parameters
    ----------
    ax : GeoAxes
        GeoAxes from cartopy
    lon : float
        longitude
    lat : float
        latitude

    Returns
    -------
    (x, y) 
        x y in fractional axes coordinats x,y in [0, 1]
    """

    # Get xy in data coordinates
    p_data = ax.projection.transform_point(lon, lat, Geodetic())

    # Transform from data to disp
    p_disp = ax.transData.transform(p_data)

    # Transform from disp to axes coordinats
    p_axes = ax.transAxes.inverted().transform(p_disp)

    return p_axes
# External
import numpy as np
import matplotlib.pyplot as plt
from obspy import read_inventory
from cartopy.crs import PlateCarree

# Internal
from .inv2geoloc import inv2geoloc
from ..plot_util.plot_map import plot_map


def fix_extent(minlon, maxlon, minlat, maxlat):

    latb = (maxlat - minlat) * 0.05
    lonb = (maxlon - minlon) * 0.05

    # Max lat
    if maxlat + latb > 90.0:
        maxlat = 90.0
    else:
        maxlat = maxlat + latb

    # Min lat
    if minlat - latb < -90.0:
        minlat = -90.0
    else:
        minlat = minlat - latb

    # Max lon
    if maxlon + lonb > 180.0:
        maxlon = 180.0
    else:
        maxlon = maxlon + lonb

    # Minlon
    if minlon - lonb < -180.0:
        minlon = -180.0
    else:
        minlon = minlon - lonb

    return [minlon, maxlon, minlat, maxlat]


def plot_station_xml(filename: str, outputfile: str or None = None):
    """Plots station_xml to map

    Parameters
    ----------
    filename : str
        StationXML

    """

    # Get latitudes and longitudes
    lat, lon = inv2geoloc(read_inventory(filename))

    # Get aspect
    minlat, maxlat = np.min(lat), np.max(lat)
    minlon, maxlon = np.min(lon), np.max(lon)
    # Get extent
    extent = fix_extent(minlon, maxlon, minlat, maxlat)

    aspect = (extent[1] - extent[0])/(extent[3] - extent[2])

    # Plot things
    plt.figure(figsize=(aspect*4, 4))
    ax = plt.axes(projection=PlateCarree())

    plot_map()
    plt.plot(lon, lat, 'v', label="Stations", markeredgecolor='k',
             markerfacecolor=(0.8, 0.3, 0.3))

    if outputfile is not None:
        plt.savefig(outputfile)
    else:
        plt.show()

    return ax

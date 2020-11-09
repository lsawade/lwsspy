from os.path import join, dirname, abspath
from netCDF4 import Dataset


def read_etopo_ice():
    """All topography

    Returns
    -------
    Tuple
        lon, lat, topography values
    """

    # Get filename
    datadir = join(dirname(dirname(__file__)), 'data')
    filename = join(datadir, 'topography', 'etopo1', 'ETOPO1_Bed_g_gmt4.grd')

    # Get data data
    etopodata = Dataset(filename, "r", format="NETCDF4")
    lat = etopodata['y'][:]
    lon = etopodata['x'][:]
    topo = etopodata['z'][:]

    return lon, lat, topo

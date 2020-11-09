from os.path import join, dirname, abspath, exists
import xarray as xr
import pooch
import lwsspy as lpy


def read_etopo_bedrock(version='bedrock', **kwargs):
    """All topography

    Returns
    -------
    Tuple
        lon, lat, topography values

    """

    version = version.lower()
    urlmod = {'ice': 'ice_surface',
              'bedrock': 'bedrock'}
    available = {
        "ice": "ETOPO1_Ice_g_gmt4.grd.gz",
        "bedrock": "ETOPO1_Bed_g_gmt4.grd.gz",
    }
    url = (f'https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/'
           f'{urlmod[version]}/grid_registered/netcdf/{available[version]}')

    # Get filename
    datadir = join(dirname(dirname(__file__)), 'data')
    fname = join(datadir, available[version][:-3])

    if exists(fname) is False:
        lpy.download_and_unzip(url, fname)

    # Get data data
    grid = xr.open_dataset(fname, **kwargs)

    # Add more metadata and fix some names
    names = {"ice": "Ice Surface", "bedrock": "Bedrock"}
    grid = grid.rename(z=version, x="longitude", y="latitude")
    grid[version].attrs["long_name"] = "{} relief".format(names[version])
    grid[version].attrs["units"] = "meters"
    grid[version].attrs["vertical_datum"] = "sea level"
    grid[version].attrs["datum"] = "WGS84"
    grid.attrs["title"] = "ETOPO1 {} Relief".format(names[version])
    grid.attrs["doi"] = "10.7289/V5C8276M"

    return grid[version].longitude, grid[version].latitude, grid[version].version

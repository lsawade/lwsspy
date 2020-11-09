import os
from os.path import join, dirname, abspath, exists
import xarray as xr
import pooch
import lwsspy as lpy


def read_etopo(version='bedrock', **kwargs):
    """All topography

    Returns
    -------
    Tuple
        lon, lat, topography values

    """

    version = version.lower()
    names = {"ice": "Ice Surface", "bedrock": "Bedrock"}
    urlmod = {'ice': 'ice_surface',
              'bedrock': 'bedrock'}
    available = {
        "ice": "ETOPO1_Ice_g_gmt4.grd.gz",
        "bedrock": "ETOPO1_Bed_g_gmt4.grd.gz",
    }
    url = (f'https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/'
           f'{urlmod[version]}/grid_registered/netcdf/{available[version]}')

    # Get filename
    zipname = join(lpy.DOWNLOAD_CACHE, available[version])
    fname = zipname[:-3]

    # Downloading if not in download cache
    if exists(fname) is False:
        if exists(zipname) is False:
            lpy.print_action(f"Downloading ETOPO1 - {names[version]}")
            lpy.downloadfile(url, zipname)
        lpy.print_action("Unzipping...")
        lpy.ungzip(zipname, fname)

    # If only zip left remove zip
    if exists(zipname):
        lpy.print_action("Removing zipfile...")
        os.remove(zipname)

    # Get data data
    grid = xr.open_dataset(fname, **kwargs)

    # Add more metadata and fix some names
    grid = grid.rename(z=version, x="longitude", y="latitude")
    grid[version].attrs["long_name"] = f"{names[version]} relief".format()
    grid[version].attrs["units"] = "meters"
    grid[version].attrs["vertical_datum"] = "sea level"
    grid[version].attrs["datum"] = "WGS84"
    grid.attrs["title"] = f"ETOPO1 {names[version]} Relief"
    grid.attrs["doi"] = "10.7289/V5C8276M"

    return (grid[version].longitude, grid[version].latitude,
            grid[version].data)

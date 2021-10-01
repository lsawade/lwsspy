from os import remove
from os.path import join, dirname, exists
import xarray as xr

import lwsspy as lpy


def read_litho(*args, **kwargs) -> xr.Dataset:
    """Reads in the Litho model

    Returns
    -------
    xr.Dataset
        Litho Grid

    Last modified: Lucas Sawade, 2020.11.10 9.30.00 (lsawade@princeton.edu)
    """
    filename = join(lpy.DOWNLOAD_CACHE, 'LITHO1.0.nc')
    url = join(lpy.EMC_DATABASE, 'LITHO1.0.nc')

    if exists(filename) is False:
        lpy.print_action(f"Downloading: {url}")
        lpy.downloadfile(url, filename)

    # Get data data
    grid = xr.open_dataset(filename, *args, **kwargs)

    return grid

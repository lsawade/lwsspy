# External
from os import remove
from os.path import join, dirname, exists
import xarray as xr

# Internal
from .. import shell as lshell
from .. import base as lbase
from .. import utils as lutils


def read_litho(*args, **kwargs) -> xr.Dataset:
    """Reads in the Litho model

    Returns
    -------
    xr.Dataset
        Litho Grid

    Last modified: Lucas Sawade, 2021.10.04 11.30.00 (lsawade@princeton.edu)
    """
    filename = join(lbase.DOWNLOAD_CACHE, 'LITHO1.0.nc')
    url = join(lbase.EMC_DATABASE, 'LITHO1.0.nc')

    if exists(filename) is False:
        lutils.print_action(f"Downloading: {url}")
        lshell.downloadfile(url, filename)

    # Get data data
    grid = xr.open_dataset(filename, *args, **kwargs)

    return grid

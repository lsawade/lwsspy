import os
from glob import glob
import numpy as np
import netCDF4 as nc
import lwsspy as lpy


def get_slabs(key='dep'):
    """Load them if dowloaded already, download them otherwise.

    I got the URL from https://www.sciencebase.gov/catalog/item/5aa1b00ee4b0b1c392e86467

    If the hardcoded one doesn't work anymore, the URL may have changed.

    Parameters
    ----------

    key: str, optional
        which parameter to read, see table below
        =====  ===========
        key    parameter 
        =====  ===========
        'dep'  depth
        'dip'  dip
        'str'  strike
        'thk'  thickness
        'unc'  uncertainty
        =====  ===========

    Returns
    -------
    list
        of netcdf datasets.

    """

    # URL that points to tar.gz
    url = "https://www.sciencebase.gov/catalog/file/get/5aa1b00ee4b0b1c392e86467?f=__disk__d5%2F91%2F39%2Fd591399bf4f249ab49ffec8a366e5070fe96e0ba"
    tarname = os.path.join(lpy.base.DOWNLOAD_CACHE, 'slabs.tar.gz')
    dirname = os.path.join(lpy.base.DOWNLOAD_CACHE, 'slabs')

    # If dirname exists nothing needs to be done
    if not os.path.exists(dirname):

        # If zip doesn't exist, you gotta download
        if not os.path.exists(tarname):
            lpy.shell.downloadfile(url, tarname)

        # Then, either way, unzipping has to be done
        lpy.shell.untar(tarname, dirname)

    # Get depths. The extra star, because untarring creates extra dir
    slabfiles = glob(os.path.join(dirname, f'*/*{key}*'))

    # Datasets
    dss = []

    for slabfile in slabfiles:
        ds = nc.Dataset(slabfile)
        dss.append(ds)

    return dss


def get_slab_minmax(dss=None):

    # Get slabs
    if dss is None:
        dss = get_slabs()

    # Get vmin, vmax
    mins, maxs = [], []
    for ds in dss:
        mins.append(np.nanmin(ds['z'][:, :].data))
        maxs.append(np.nanmax(ds['z'][:, :].data))

    vmin = np.min(mins)
    vmax = np.max(maxs)

    return vmin, vmax


def get_slab_extend(ds):

    # Get vmin, vmax
    xmin = np.nanmin(ds['x'][:].data)
    xmax = np.nanmax(ds['x'][:].data)
    ymin = np.nanmin(ds['y'][:].data)
    ymax = np.nanmax(ds['y'][:].data)

    return xmin, xmax, ymin, ymax

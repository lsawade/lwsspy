import numpy as np
import matplotlib.pyplot as plt
from .. import geo as lgeo
from cartopy import crs


def plot_slabs(
        *args, plot_levels: bool = False, levels=None, ax=None,
        dss=None, cargs=tuple(), ckwargs=dict(), **kwargs):
    """Plots the slabs globally. args and kwargs are propagated to contourf.
    Use cargs, and ckwargs for contour lines ()

    Parameters
    ----------
    plot_levels : bool, optional
        [description], by default False
    levels : iterable, optional
        cotour levels, by default None
    ax : mpl.Axes, optional
        axes, by default None
    dss : list, optional
        list of netcdf datasets, by default None
    cargs : tuple, optional
        contour arguments, by default tuple()
    ckwargs : dict, optional
        contour keywords, by default dict()
    """

    # get slabs
    if dss is None:
        dss = lgeo.get_slabs()

    # Create axes if needed
    if ax is None:
        ax = plt.gca()

    # Get min and max
    if levels is None:
        vmin, vmax = lgeo.get_slab_minmax(dss=dss)
    else:
        vmin = None
        vmax = None

    # Plot contours
    for ds in dss:
        lon = ds['x'][:].data
        lat = ds['y'][:].data
        z = ds['z'][:, :].data
        llon, llat = np.meshgrid(lon, lat)

        plt.contourf(llon, llat, z, *args, levels=levels,
                     transform=crs.PlateCarree(), **kwargs)

        if plot_levels:
            plt.contour(llon, llat, z, *cargs, levels=levels,
                        vmin=vmin, vmax=vmax,
                        transform=crs.PlateCarree(), **ckwargs)

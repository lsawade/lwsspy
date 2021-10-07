# External imports
from typing import List
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy as np

# Local
from .. import maps as lmaps
from .. import plot as lplot


def plot_topography(extent: List[float] = [-180.0, 180.0, -90.0, 90.0],
                    dataset: str = 'etopo1', subsampling: int = 8,
                    sigma: float = 0.0, colorbar: bool = True):
    """Plots topography into existing axes. Much faster if extent is set.

    Parameters
    ----------
    extent : List[float], optional
        Map extent, by default [-180.0, 180.0, -90.0, 90.0]
    dataset : str, optional
        which topography map is used, by default 'etopo1' and only supported
        one right now.
    subsampling : int, optional
        number by which to decimate the topographic map, default 8
    sigma : float, optional
        standard devation for the gaussian image filter, default 0.0
        5.0-10.0 for a cartoonish look
    colorbar: bool, optional
        whether to draw the elevation colrbar or not.
    """

    # Get topography
    etopo_bed = lmaps.read_etopo()
    # etopo_ice = lmaps.read_etopo(version='ice')

    # decimate
    fextent = lmaps.fix_map_extent(extent, fraction=0.1)
    aspect = (fextent[1] - fextent[0])/(fextent[3] - fextent[2])
    grid_bed = etopo_bed.sel(latitude=slice(fextent[2], fextent[3]),
                             longitude=slice(fextent[0], fextent[1]))
    # grid_ice = etopo_ice.sel(latitude=slice(fextent[2], fextent[3]),
    #                          longitude=slice(fextent[0], fextent[1]))

    # Get current axis
    ax = plt.gca()

    # Illuminate the scene from the northwest
    ls = LightSource(azdeg=270, altdeg=45)
    ve = 1.0

    cmap = lmaps.topocolormap()
    norm = lplot.FixedPointColorNorm(vmin=-10000, vmax=8000)

    # Interpolating 1000x aspect
    mat = gaussian_filter(
        grid_bed.bedrock[::-subsampling, ::subsampling].values,
        sigma=sigma, order=0, mode=['nearest', 'wrap'])
    bed = ax.imshow(ls.shade(mat, vert_exag=ve, blend_mode='soft',
                             cmap=cmap, norm=norm), extent=fextent,
                    zorder=-15)
    # ice = ax.imshow(grid_ice.ice[::-1, :], cmap='Blues',
    #                 extent=fextent,
    #                 zorder=-14)
    if colorbar:
        cbar = lplot.nice_colorbar(ScalarMappable(
            cmap=cmap, norm=norm), pad=0.125/aspect)
        cbar.set_label("Elevation [m]")

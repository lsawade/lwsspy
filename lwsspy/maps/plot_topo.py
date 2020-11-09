# External imports
from typing import List
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import lwsspy as lpy

import numpy as np


def plot_topo(extent: List[float] = [-180.0, 180.0, -90.0, 90.0],
              dataset: str = 'etopo1'):

    # Get topography
    lon, lat, topo = lpy.read_etopo_bedrock()

    # Get current axis
    ax = plt.gca()

    # Create new vectors to interpolate
    fextent = lpy.fix_map_extent(extent)
    # olon, olat = np.meshgrid(lon, lat)

    print(lon.shape, lat.shape, topo.shape)
    print(np.min(topo), np.max(topo))

    # Decimating
    lonpos = np.where(((fextent[0] <= lon)
                       & (lon <= fextent[1])))[0]
    latpos = np.where(((fextent[2] <= lat)
                       & (lat <= fextent[3])))[0]
    minrow, maxrow = np.min(latpos), np.max(latpos)
    mincol, maxcol = np.min(lonpos), np.max(lonpos)

    # Interpolating 1000x aspect
    aspect = (fextent[1] - fextent[0])/(fextent[3] - fextent[2])
    # olat = olat[rows, cols]
    # olon = olon[rows, cols]
    # topo = topo[rows, cols]

    # Plot
    # olon, olat = np.meshgrid(, olat)

    # ax.pcolormesh(olon[minrow:maxrow+1, mincol:maxcol+1],
    #               olat[minrow:maxrow+1, mincol:maxcol+1],
    #               topo[minrow:maxrow+1, mincol:maxcol+1],
    #               cmap='terrain')
    im = ax.imshow(topo, cmap='terrain')
    plt.colorbar(im)

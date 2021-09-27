from math import e
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cartopy.crs import Mollweide, PlateCarree
import numpy as np
from .slabkdtree import SlabKDTree
from ..maps.plot_map import plot_map
from ..maps.gctrack import gctrack
from ..plot_util.nice_colorbar import nice_colorbar


def plot_south_america():

    # Some point at the coast
    olat, olon = -22.336766, -70.042413

    # Latitude slices to plot
    lat = np.arange(2.5, -42.5, -10)
    lon0 = np.linspace(-85, -80, len(lat))
    lon1 = np.linspace(-65, -55, len(lat))
    lon0[0] += 2.5
    lon1[0] += 2.5
    lon0[2] += 2.5
    lon1[2] += 2.5
    lon0[3] += 2.5
    lon1[3] += 2.5
    N = len(lat)

    # Get SlabKDTree (Large, takes some time if not buffered)
    skt = SlabKDTree()

    # Check if location in slab list
    slab_list = []
    for _i, _bbox in enumerate(skt.bbox_list):

        # Check if in bbox
        check = np.any(skt.checkifinbbox(_bbox, olat, olon))

        if check:
            slab_list.append(_i)

    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(nrows=N, ncols=2, )  # width_ratios=[1, 2])

    mapax = fig.add_subplot(gs[:, 0], projection=Mollweide())
    plot_map(ax=mapax, lw=0.0)
    mapax.set_extent([-95, -45, -55, 20])

    for _slab in slab_list:

        slabimg = plt.pcolormesh(
            skt.lon_list[_slab], skt.lat_list[_slab],
            -skt.data_list[_slab]['dep'], transform=PlateCarree(),
            cmap='rainbow_r', shading='auto'
        )

        nice_colorbar(orientation='horizontal', fraction=0.025, pad=0.025)

    axes = []
    factors0 = {0: 0.0, 1: 0.0, 2: -2.5, 3: 0.0, 4: 0.0}
    factors1 = {0: -2.5, 1: 2.5, 2: 5.0, 3: 5.0, 4: 5.0}

    # GCTracks
    tracks = []

    for _i, (_lat, _lon0, _lon1) in enumerate(zip(lat, lon0, lon1)):

        lats, lons, dists = gctrack(
            [_lat + factors0[_i], _lat + factors1[_i]], [_lon0, _lon1], dist=0.075)

        tracks.append((lats, lons, dists))

        mapax.plot(lons, lats, 'k', markeredgecolor='k',
                   transform=PlateCarree())
        mapax.plot(lons[0], lats[0], 'ok', markeredgecolor='k',
                   transform=PlateCarree())
        mapax.plot(lons[-1], lats[-1], 'ow',
                   markeredgecolor='k', transform=PlateCarree())

        ax = fig.add_subplot(gs[_i, 1])
        axes.append(ax)
        ax.plot(0, 0, 'ok', markeredgecolor='k', clip_on=False, zorder=10)
        ax.plot(17.5, 0, 'ow', markeredgecolor='k', clip_on=False, zorder=10)

        if _i == N-1:
            ax.set_xlabel('Offset [deg]')
        else:
            ax.set_xticklabels([])

        interps, data_list = skt.interpolate(lats, lons)

        for _slab in slab_list:

            ax.fill_between(
                dists, -(data_list[_slab]['dep'] - data_list[_slab]['thk']),
                -(data_list[_slab]['dep']), fc='lightgray')

        ax.set_xlim([0, 17.5])
        ax.set_ylim([0, 700])
        ax.invert_yaxis()

    return tracks, mapax, axes

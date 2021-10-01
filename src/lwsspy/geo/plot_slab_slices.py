from math import dist, e
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from cartopy.crs import Mollweide, PlateCarree
import numpy as np
from numpy.lib import meshgrid
from .slabkdtree import SlabKDTree
from ..maps.plot_map import plot_map
from ..maps.gctrack import gctrack
from ..plot.nice_colorbar import nice_colorbar
from ..plot.axes_from_axes import axes_from_axes
from ..plot.plot_label import plot_label
from ..constants import abc

tonga_dict = dict(
    # Some point at the coast
    olat=-22.336766,
    olon=-180.042413,
    lat0=[-15.4, -20.5, -26.0, -29.56],
    lon0=[-173.0, -173.75, -176.0, -176.6],
    lat1=[-17.35, -20.0, -25.0, -30.753],
    lon1=[-179.0, -178.5, -179.9, 179.84],
    xlim=[0, 6.5],
    ylim=[0, 400],
    dist=0.0025,
    mapextent=[167.5, -170 + 360, -50, -13],
    caxextent=[1-0.15, 0.0, 0.03, 0.3],
    central_longitude=180.0,
    invertx=True,
    c180=True,
)


def plot_slab_slices(slice_dict: dict = tonga_dict):

    # Latitude slices to plot
    olat = slice_dict['olat']
    olon = slice_dict['olon']
    lat0 = slice_dict['lat0']
    lat1 = slice_dict['lat1']
    lon0 = slice_dict['lon0']
    lon1 = slice_dict['lon1']
    xlim = slice_dict['xlim']
    ylim = slice_dict['ylim']
    dist = slice_dict['dist']
    mapextent = slice_dict['mapextent']
    caxextent = slice_dict['caxextent']
    central_longitude = slice_dict['central_longitude']
    invertx = slice_dict['invertx']
    c180 = slice_dict['c180']
    N = len(lat0)

    # Get SlabKDTree (Large, takes some time if not buffered)
    skt = SlabKDTree(rebuild=False)

    # Check if location in slab list
    slab_list = []

    # Create checkspace
    blat = np.linspace(mapextent[2], mapextent[3], 50)
    blon = np.linspace(mapextent[0], mapextent[1], 50)
    bblon, bblat = np.meshgrid(blon, blat)

    for _i, _bbox in enumerate(skt.bbox_list):

        # Check if in bbox
        check = np.any(skt.checkifinbbox(
            _bbox, bblat.flatten(), bblon.flatten()))
        # print(_i, check)
        if check:
            slab_list.append(_i)

    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(nrows=1, ncols=2, wspace=0.05, width_ratios=[1, 1.25])
    gs_slice = GridSpecFromSubplotSpec(
        nrows=N, ncols=1,
        subplot_spec=gs[1])

    mapax = fig.add_subplot(gs[:, 0], projection=Mollweide(
        central_longitude=central_longitude))
    plot_map(ax=mapax, lw=0.0)
    plot_map(ax=mapax, fill=False, outline=True, zorder=10, lw=0.0)
    mapax.set_extent(mapextent)
    plot_label(mapax, f'{abc[0]})', location=5, dist=0.025, box=False)

    # Normalize so that all slabs have the same colorspace
    norm = Normalize(vmin=0.0, vmax=600.0)
    for _slab in slab_list:
        tmp_lon = np.where(
            skt.lon_list[_slab] < 0.0, skt.lon_list[_slab] + 360.0, skt.lon_list[_slab])

        slabimg = plt.pcolormesh(
            tmp_lon, skt.lat_list[_slab],
            -skt.data_list[_slab]['dep'], transform=PlateCarree(),
            cmap='rainbow_r', shading='auto', norm=norm)

    cax = axes_from_axes(mapax, 999, caxextent)
    nice_colorbar(cax=cax, orientation='vertical', label='Depth [km]')
    cax.invert_yaxis()
    cax.tick_params(labelleft=True, labelright=False, left=True, right=False,
                    which='both')

    # Collecting the computed thigns
    axes = []
    tracks = []
    mcmap = plt.get_cmap('gray')
    mcolors = mcmap(np.linspace(0.0, 1.0, N))

    for _i, (_lat0, _lat1, _lon0, _lon1) in enumerate(zip(lat0, lat1, lon0, lon1)):

        lats, lons, dists = gctrack(
            [_lat0, _lat1], [_lon0, _lon1], dist=dist)

        if c180 is True:
            lons = np.where(lons < 0.0, lons + 360.0, lons)
        tracks.append((lats, lons, dists))

        mapax.plot(lons, lats, 'k', markeredgecolor='k',
                   transform=PlateCarree(), zorder=20)
        mapax.plot(lons[0], lats[0], 's', markeredgecolor='k',
                   markerfacecolor=mcolors[_i],
                   transform=PlateCarree(), zorder=20)
        mapax.plot(lons[-1], lats[-1], 'sw',
                   markeredgecolor='k', transform=PlateCarree(), zorder=20)

        if _i != 0:
            sharex = axes[_i-1]
            sharey = axes[_i-1]
        else:
            sharex = None
            sharey = None

        ax = fig.add_subplot(gs_slice[_i], sharex=sharex, sharey=sharey)
        axes.append(ax)
        ax.plot(0, ylim[0], 's', markeredgecolor='k',
                markerfacecolor=mcolors[_i],
                clip_on=False, zorder=10)
        ax.plot(dists[-1], ylim[0], 'sw', markeredgecolor='k',
                clip_on=False, zorder=10)
        plot_label(ax, f'{abc[1+_i]})', location=5, dist=0.025, box=False)
        ax.set_ylabel('Depth [km]')
        if _i == N-1:
            ax.set_xlabel('Offset [deg]')
        else:
            ax.label_outer()

        interps, data_list = skt.interpolate(lats, lons)

        for _slab in slab_list:

            if data_list[_slab] is not None:
                ax.fill_between(
                    dists, -(data_list[_slab]['dep'] -
                             data_list[_slab]['thk']),
                    -(data_list[_slab]['dep']), fc='lightgray')
                ax.plot(dists, -(data_list[_slab]
                                 ['dep'] - data_list[_slab]['unc']),
                        '--', lw=0.5, c='gray')
                ax.plot(dists, -(data_list[_slab]
                                 ['dep'] + data_list[_slab]['unc']),
                        '--', lw=0.5, c='gray')
                ax.plot(dists, -(data_list[_slab]['dep']),
                        '-', lw=0.5, c='black')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelleft=False, labelright=True)
        ax.yaxis.set_label_position("right")

        ax.invert_yaxis()
        if invertx is True:
            ax.invert_xaxis()

    return tracks, mapax, axes, skt, slab_list

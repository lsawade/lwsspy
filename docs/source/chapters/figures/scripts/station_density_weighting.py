import os
from cartopy.crs import PlateCarree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
import matplotlib.ticker as ticker
from matplotlib import gridspec


import lwsspy as lpy
lpy.updaterc()


def plot_station_density(lat, lon, condict=dict(ctype='fracmax', param=0.33),
                         axlist=None):
    # Create GeoWeights class
    gw = lpy.GeoWeights(lat, lon)
    vref, vcond, ref, cond = gw.get_condition(**condict)
    if axlist is None:
        fig = plt.figure(figsize=(11.5, 3.5))
        plt.subplots_adjust(wspace=0.025, bottom=0.125, left=0.1, right=0.95)
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.3, 0.70], figure=fig)
        ax1 = plt.subplot(gs[0, 0])
        lpy.plot_label(ax1, 'a)', location=6, box=False)
        ax2 = plt.subplot(gs[0, 1], projection=PlateCarree())
        lpy.plot_label(ax2, 'b)', location=6, box=False)
    else:
        ax1, ax2 = axlist

    # Plot condition number stuff
    plt.sca(ax1)
    plt.plot([0, 180], [cond, cond], 'k--')
    plt.plot([ref, ref], [0, 100], 'k--')
    plt.plot(vref, vcond, 'k.-',
             label=r"$\mathrm{cond}(\mathbf{\omega}(\Delta_0))$")
    plt.plot(ref, cond, 'k*', markersize=15,
             label="Choice")
    plt.xlim(0, 180)
    plt.ylim(0, 1.2*np.max(vcond))
    plt.xlabel('Reference $\Delta$')
    plt.ylabel('Condition Number - 1')
    plt.legend(frameon=False, loc='upper right')

    if condict['ctype'] == 'fracmax':
        title = f"{condict['param']:3.2f} of the Maximum"
    elif condict['ctype'] == 'q':
        title = f"Q1 - {condict['param']:3.2f} Quantile"
    elif condict['ctype'] == 'max':
        title = 'Maximum of Cond. dist.',
    elif condict['ctype'] == 'dist':
        title = f"{condict['param'].capitalize()} of $\Delta_{{ij}}$ Dist."
    else:
        raise ValueError(f"{condict['ctype']} not implemented.")
    plt.title(title)

    # Compute actual weight
    weights = gw.get_weights(ref=ref)

    # Plot weighs on a map number stuff
    plt.sca(ax2)
    lpy.plot_map()
    lpy.remove_ticklabels(ax2)
    plt.scatter(lon, lat, c=weights, cmap='rainbow',
                norm=LogNorm(vmin=min(weights), vmax=max(weights)),
                transform=PlateCarree())
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    # formatter = LogFormatter(10, labelOnlyBase=False)

    formatter = ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
    cb = lpy.nice_colorbar(orientation='vertical', ticks=[0.3, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0],  # np.arange(0.3, 3.0, 0.3),
                           format=formatter, aspect=40, pad=0.025)
    cb.set_label('Weight')
    plt.title(
        f"min: {np.min(weights):4.2f}   "
        f"max: {np.max(weights):4.2f}   "
        f"median: {np.median(weights):4.2f}   "
        f"mean: {np.mean(weights):4.2f} "
    )


# Create some different even2D points to create artificial station densities.
lat_eur = 53
lon_eur = 23
lat_us = 0
lon_us = -90
width = 360
height = 180
x, y = lpy.even2Dpoints(100, width, height, 10)
x1, y1 = lpy.even2Dpoints(50, 60, 40, 1)
x2, y2 = lpy.even2Dpoints(50, 50, 120, 3)

# Combine
lat = np.hstack((np.array(y), np.array(y1)+lat_eur, np.array(y2)+lat_us))
lon = np.hstack((np.array(x), np.array(x1)+lon_eur, np.array(x2)+lon_us))
# lat = np.hstack((np.array(y), np.array(y1)+lat_eur))
# lon = np.hstack((np.array(x), np.array(x1)+lon_eur))


fig = plt.figure(figsize=(9.5, 11))
plt.subplots_adjust(
    wspace=0.025, hspace=0.25, left=0.1, right=0.95, bottom=0.05, top=0.95)
gs = gridspec.GridSpec(4, 2, width_ratios=[0.3, 0.70], figure=fig)

labels = [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]
cdicts = [
    dict(ctype='fracmax', param=0.33),
    dict(ctype='q', param=0.1),
    dict(ctype='dist', param='median'),
    dict(ctype='dist', param='mean'),
]
for i in range(4):
    ax1 = plt.subplot(gs[i, 0])
    lpy.plot_label(ax1, f'{labels[i][0]})', location=6, box=False)
    ax2 = plt.subplot(gs[i, 1], projection=PlateCarree())
    lpy.plot_label(ax2, f'{labels[i][1]})', location=6, box=False)
    plot_station_density(lat, lon, condict=cdicts[i], axlist=[ax1, ax2])
    if i != 3:
        ax1.set_xlabel('')
        lpy.remove_xticklabels(ax1)

plt.savefig(os.path.join(lpy.DOCFIGURES, "station_density_weighting_even.pdf"))
plt.savefig(os.path.join(lpy.DOCFIGURES, "station_density_weighting_even.svg"))


x, y = lpy.even2Dpoints(1, width, height, 10)
x1, y1 = lpy.even2Dpoints(100, 60, 40, 1)

# Combine
lat = np.hstack((np.array(y), np.array(y1)+lat_eur))
lon = np.hstack((np.array(x), np.array(x1)+lon_eur))

fig = plt.figure(figsize=(9.5, 11))
plt.subplots_adjust(
    wspace=0.025, hspace=0.25, left=0.1, right=0.95, bottom=0.05, top=0.95)
gs = gridspec.GridSpec(4, 2, width_ratios=[0.3, 0.70], figure=fig)

labels = [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]
cdicts = [
    dict(ctype='fracmax', param=0.33),
    dict(ctype='q', param=0.1),
    dict(ctype='dist', param='median'),
    dict(ctype='dist', param='mean'),
]
for i in range(4):
    ax1 = plt.subplot(gs[i, 0])
    lpy.plot_label(ax1, f'{labels[i][0]})', location=6, box=False)
    ax2 = plt.subplot(gs[i, 1], projection=PlateCarree())
    lpy.plot_label(ax2, f'{labels[i][1]})', location=6, box=False)
    plot_station_density(lat, lon, condict=cdicts[i], axlist=[ax1, ax2])
    if i != 3:
        ax1.set_xlabel('')
        lpy.remove_xticklabels(ax1)

plt.savefig(os.path.join(lpy.DOCFIGURES,
                         "station_density_weighting_uneven.pdf"))
plt.savefig(os.path.join(lpy.DOCFIGURES,
                         "station_density_weighting_uneven.svg"))


plt.show()

# plt.switch_backend('pdf')
# plt.savefig(os.path.join(lpy.DOCFIGURES, "station_density_weighting.pdf"))
# plt.savefig(os.path.join(lpy.DOCFIGURES, "station_density_weighting.svg"))

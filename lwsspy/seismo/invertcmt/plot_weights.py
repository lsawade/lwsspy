import _pickle as cPickle
import matplotlib.pyplot as plt
import lwsspy as lpy
from cartopy.crs import PlateCarree, AzimuthalEquidistant
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import numpy as np


def plot_single_weight_set(ax, lat, lon, weights, nomean=False):

    lpy.plot_map()

    # Plot nonnormalized weights
    if nomean:
        norm = LogNorm(vmin=np.min(weights), vmax=np.max(weights))
        cmap = "rainbow"
    else:
        norm = lpy.MidPointLogNorm(vmin=np.min(weights), vmax=np.max(weights),
                                   midpoint=1.0)
        cmap = "RdBu_r"

    plt.scatter(lon, lat, c=weights, cmap=cmap,
                norm=norm,
                edgecolors='k', linewidths=0.5,
                transform=PlateCarree())
    formatter = ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
    cb = lpy.nice_colorbar(orientation='horizontal',
                           # np.arange(0.3, 3.0, 0.3),
                           ticks=[0.3, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0],
                           format=formatter, aspect=40, pad=0.075)
    cb.set_label("Weights")
    if nomean:
        lpy.plot_label(
            ax,
            f"min: {np.min(weights):7.4f}\n"
            f"max: {np.max(weights):7.4f}\n"
            f"sum: {np.sum(weights):7.4f}\n",
            f"mean: {np.mean(weights):7.4f}\n"
            f"median: {np.median(weights):7.4f}\n",
            location=3, box=True, dist=-0.1, fontdict=dict(fontsize='small'))
    else:
        lpy.plot_label(
            ax,
            f"min: {np.min(weights):3.2f}\n"
            f"max: {np.max(weights):3.2f}\n"
            f"mean: {np.mean(weights):3.2f}\n"
            f"median: {np.median(weights):3.2f}\n",
            location=3, box=True, dist=-0.1, fontdict=dict(fontsize='small'))


def plot_weightpickle(weightpickle: str):

    with open(weightpickle, "rb") as f:
        weights = cPickle.load(f)

    plot_weights(weights)


def plot_weights(weights: dict):

    # Weights to be plotted
    component_list = ["Z", "R", "T"]
    weightlist = ["geographical", "azimuthal", "combination", "final"]

    # Event location
    lat0, lon0 = weights["event"]

    for _wtype in weights.keys():
        if _wtype == "event":
            continue
        # Get wave weight
        waveweight = weights[_wtype]["weight"]

        # Create Base figure
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, len(weightlist), figure=fig)
        plt.subplots_adjust(bottom=0.025, top=0.925, left=0.05, right=0.95)

        for _i, _component in enumerate(component_list):
            latitudes = weights[_wtype][_component]["lat"]
            longitudes = weights[_wtype][_component]["lon"]

            for _j, _weight in enumerate(weightlist):

                # Create axes
                ax = plt.subplot(gs[_i, _j], projection=AzimuthalEquidistant(
                    central_longitude=lon0, central_latitude=lat0))
                ax.set_global()

                # Get weights
                plotweights = weights[_wtype][_component][_weight]

                # Plot weights
                if _weight == "final":
                    nomean = True
                else:
                    nomean = False

                plot_single_weight_set(
                    ax, latitudes, longitudes, plotweights, nomean=nomean)

                if _i == 0:
                    lpy.plot_label(ax, _weight.capitalize(),
                                   location=14, box=False, dist=0.05)
                if _j == 0:
                    lpy.plot_label(ax, _component.capitalize(),
                                   location=13, box=False, dist=0.05)

        plt.suptitle(f"{_wtype.capitalize()}: {waveweight:6.4f}")
        plt.savefig(f"./weights_{_wtype}.pdf")

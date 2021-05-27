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
            f"median: {np.median(weights):7.4f}",
            location=3, box=True, dist=-0.1, fontdict=dict(fontsize='small'))
    else:
        lpy.plot_label(
            ax,
            f"min: {np.min(weights):3.2f}\n"
            f"max: {np.max(weights):3.2f}\n"
            f"mean: {np.mean(weights):3.2f}\n"
            f"median: {np.median(weights):3.2f}",
            location=3, box=True, dist=-0.1, fontdict=dict(fontsize='small'))


def plot_single_weight_hist(ax, weights, nbins=10, color="lightgray"):

    weights_norm = weights/np.min(weights)
    nb, _, _ = plt.hist(weights_norm,
                        bins=nbins,
                        edgecolor='k',
                        facecolor=color,
                        linewidth=0.75,
                        linestyle='-',
                        histtype='stepfilled')

    lpy.plot_label(
        ax,
        f"min: {np.min(weights_norm):7.4f}\n"
        f"max: {np.max(weights_norm):7.4f}\n"
        f"sum: {np.sum(weights_norm):7.4f}\n"
        f"mean: {np.mean(weights_norm):7.4f}\n"
        f"median: {np.median(weights_norm):7.4f}",
        location=7, box=True, dist=-0.1, fontdict=dict(fontsize='small'))


def plot_weightpickle(weightpickle: str):

    with open(weightpickle, "rb") as f:
        weights = cPickle.load(f)

    # plot_weights(weights)
    plot_weight_histograms(weights)


def plot_weight_histograms(weights: dict):

    # Weights to be plotted
    component_list = ["Z", "R", "T"]
    weightlist = ["geographical", "azimuthal", "combination", "final"]

    for _j, _weight in enumerate(weightlist):

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig)
        plt.subplots_adjust(bottom=0.075, top=0.925, left=0.075, right=0.925)

        counter = 0
        for _i, _wtype in enumerate(weights.keys()):
            if _wtype == "event":
                continue

            # Get wave weight
            waveweight = weights[_wtype]["weight"]

            for _j, _component in enumerate(component_list):

                # Create axes
                ax = plt.subplot(gs[counter, _j])

                # Get weights
                plotweights = weights[_wtype][_component][_weight]

                # Plot histogram
                plot_single_weight_hist(ax, plotweights, nbins=10)

                if _j == 0:
                    # lpy.plot_label(ax, _wtype.capitalize() + f": {waveweight}",
                    #                location=14, box=False, dist=0.05)

                    plt.ylabel(_wtype.capitalize() + f": {waveweight:4.2f}")
                if counter == 2:
                    # lpy.plot_label(ax, _component.capitalize(),
                    #                location=13, box=False, dist=0.05)
                    plt.xlabel(_component.capitalize())

            counter += 1

        plt.suptitle(f"{_weight.capitalize()}")
        plt.savefig(f"./weights_{_weight}_histogram.pdf")


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

from typing import Callable, List, Union, Tuple
import lwsspy as lpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.crs import PlateCarree
import numpy as np


def plot_quakes(latitude, longitude, depth, moment,
                ax: Union[matplotlib.axes.Axes, None] = None,
                levels: Union[List[float], None] = None,
                cmap: str = 'jet',
                sizefunc: Callable = lambda x: (x-(np.min(x)-1))**2.5,
                legend: bool = True,
                xoffsetlegend: float = 0.0,
                yoffsetlegend: float = 0.0,
                yoffsetlegend2: float = 0.0) -> Tuple[
                    matplotlib.collections.PathCollection,
                    matplotlib.axes.Axes,
                    Union[None, matplotlib.legend.Legend],
                    Union[None, matplotlib.legend.Legend]]:
    """Plots a set of earthquakes and their basic parameters on a map with
    optional legends.

    Parameters
    ----------
    latitude : np.ndarray
        Latitudes   
    longitude : np.ndarray
        Longitudes
    depth : np.ndarray
        Depths
    moment : np.ndarray
        Moments
    ax : Union[matplotlib.axes.Axes, None], optional
        Axes to plot the earthquakes into. If None is given a figure and axes
        will be created, by default None
    levels : Union[List[float], None], optional
        List of boundaries for coloring the depths, by default None
    cmap : str, optional
        by default ``jet``, by default None
    sizefunc : Callable, optional
        Function to resize the scatter points, 
        by default lambda x : (x - (np.min(x) - 1)) ** 2.5
    legend : bool, optional
        flag on whether to plot the legend, by default True
    xoffsetlegend : float
        offsets the legend in normalized axes coordinates
    yoffsetlegend : float
        offsets the legend in normalized axes coordinates

    Returns
    -------
    Tuple[ matplotlib.collections.PathCollection, matplotlib.axes.Axes, Union[None, matplotlib.legend.Legend], Union[None, matplotlib.legend.Legend]]
        If legend is true (scatter, ax, depthlegend, momentlegend), else:
        (scatter, ax, None, None)

    """

    # Set default levels.
    if levels is None:
        # Get colors
        levels = [0.0, 10.0, 11.0, 12.5, 15.0, 20.0, 25.0,
                  50.0, 70.0, 200.0, 400.0, 600.0, 700.0]

    isort = np.argsort(depth)[::-1]

    # Create
    colormap = plt.get_cmap(cmap)
    colors = lpy.pick_colors_from_cmap(len(levels), colormap)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, len(colors), clip=True)

    # Create figure if no axes was given
    if ax is None:
        plt.figure(figsize=(9, 5.25))
        plt.subplots_adjust(left=0.025, right=0.975, bottom=0.15, top=0.95)
        ax = lpy.map_axes(proj="moll", central_longitude=-150.0)
        lpy.plot_map(zorder=-1)

    # Fix moments
    rmoments = np.floor(moment)
    unique_moments = np.unique(rmoments)
    scatter = plt.scatter(longitude[isort], latitude[isort],
                          sizefunc(rmoments[isort]), c=depth[isort],
                          transform=PlateCarree(), cmap=cmap, alpha=1.0,
                          norm=norm, edgecolor='k', linewidth=0.1)

    if legend:

        # Set legend fontsizes
        legendfontsize = "x-small"
        title_fontsize = "small"

        # Get depth legend
        handles, labels = scatter.legend_elements(num=levels)

        # Set labels
        labels = [f"{int(np.round(levels[_i]))} - {int(np.round(levels[_i + 1]))} km"
                  for _i in range(len(labels))]

        # Plot Depth legend
        legend1 = ax.legend(
            handles, labels, ncol=3, loc="upper left", title='Depth',
            bbox_to_anchor=(0.0 + xoffsetlegend, yoffsetlegend), handletextpad=0.2,
            frameon=False, fontsize=legendfontsize,
            title_fontsize=title_fontsize,
            bbox_transform=ax.transAxes)
        lpy.right_align_legend(legend1)
        ax.add_artist(legend1)

        # Get Size props of the legend entries.
        handles, _ = scatter.legend_elements(
            num=unique_moments.size, prop="sizes", alpha=0.6)

        # Create labels
        labels = [f"{_m:>4.1f} - {_m + 1:>4.1f}" for _m in unique_moments]

        # Plot Moment legend
        legend2 = ax.legend(
            handles, labels, loc="upper right", title="$M_w$", frameon=False,
            bbox_to_anchor=(1.0 + xoffsetlegend, yoffsetlegend+yoffsetlegend2),
            ncol=1, handletextpad=0.2,
            fontsize=legendfontsize, title_fontsize=title_fontsize,
            bbox_transform=ax.transAxes)
        lpy.right_align_legend(legend2)

        return scatter, ax, legend1, legend2
    else:
        return scatter, ax, None, None

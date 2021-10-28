import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.crs import PlateCarree, Mollweide

# steps = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
#          1, 1.5, 2, 2.5, 5, 10, 15, 20, 25, 30, 45]
steps = [1, 1.5, 1.8, 2, 3, 6, 10]


def plot_map(fill=True, zorder=None, labelstopright: bool = True,
             labelsbottomleft: bool = True, borders: bool = False,
             rivers: bool = False, lakes: bool = False, outline: bool = False,
             ax=None, lw=0.5):
    """Plots map into existing axes.

    Parameters
    ----------
    fill : bool, optional
        fills the continents in light gray, by default True
    zorder : int, optional
        zorder of the map, by default -10
    projection : cartopy.crs.projection, optional
        projection to be used for the map.
    labelstopright : bool, optional
        flag to turn on or off the ticks
    labelsbottomleft : bool, optional
        flag to turn on or off the ticks
    borders : bool
        plot borders. Default True
    rivers : bool
        plot rivers. Default False
    lakes : bool 
        plot lakes. Default True
    lw : float
        outline width

    Returns
    -------
    matplotlib.pyplot.Axes
        Axes in which the map was plotted

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.09.22 11.45


    """

    if ax is None:
        ax = plt.gca()

    # Change outline width
    ax.spines['geo'].set_linewidth(lw)

    if outline:
        edgecolor = 'black'
    else:
        edgecolor = 'none'

    # Add land
    if fill:
        ax.add_feature(cartopy.feature.LAND, zorder=zorder, edgecolor=edgecolor,
                       linewidth=0.5, facecolor=(0.8, 0.8, 0.8))
    else:
        ax.add_feature(cartopy.feature.LAND, zorder=zorder, edgecolor=edgecolor,
                       linewidth=0.5, facecolor=(0, 0, 0, 0))

    if borders:
        ax.add_feature(cartopy.feature.BORDERS,
                       zorder=None if zorder is None else zorder + 1,
                       facecolor='none', edgecolor=(0.5, 0.5, 0.5),
                       linewidth=0.25)

    if rivers:
        ax.add_feature(cartopy.feature.RIVERS, zorder=zorder,
                       edgecolor=(0.3, 0.3, 0.7),)
        #    edgecolor=(0.5, 0.5, 0.7) )

    if lakes:
        ax.add_feature(cartopy.feature.LAKES,
                       zorder=None if zorder is None else zorder + 1,
                       edgecolor='black', linewidth=0.5,
                       facecolor=(1.0, 1.0, 1.0))
    return ax

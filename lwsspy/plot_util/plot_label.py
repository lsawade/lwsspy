import matplotlib
import matplotlib.pyplot as plt
import lwsspy as lpy


def plot_label(ax: matplotlib.axes.Axes, label: str, aspect: float = 1,
               location: int = 1, dist: float = 0.025, box: bool = True,
               fontdict: dict = {}):
    """Plots label one of the corners of the plot.

    Plot locations are set as follows::

           6        7 
            --------
         5 |1      2|  8
           |        |
        12 |3      4|  9
            --------
          11       10


    Parameters
    ----------
    label : str
        label
    aspect : float, optional
        aspect ratio length/height, by default 1.0
    location : int, optional
        corner as described by above code figure, by default 1
    aspect : float, optional
        aspect ratio length/height, by default 0.025
    box : bool
        plots bounding box st. the label is on a background, default true

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.01.26 18.30

    """
    if box:
        boxdict = {'facecolor': 'w', 'edgecolor': 'k'}
    else:
        boxdict = {'facecolor': 'none', 'edgecolor': 'none'}

    # Get aspect of the axes
    aspect = 1.0/lpy.get_aspect(ax)

    # Inside
    if location == 1:
        plt.text(dist, 1.0 - dist * aspect, label, horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict)
    elif location == 2:
        plt.text(1.0 - dist, 1.0 - dist * aspect, label,
                 horizontalalignment='right', verticalalignment='top',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict)
    elif location == 3:
        plt.text(dist, dist * aspect, label, horizontalalignment='left',
                 verticalalignment='bottom', transform=ax.transAxes,
                 bbox=boxdict, fontdict=fontdict)
    elif location == 4:
        plt.text(1.0 - dist, dist * aspect, label,
                 horizontalalignment='right', verticalalignment='bottom',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict)
    # Outside
    elif location == 5:
        plt.text(dist, 1.0, label, horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict)
    elif location == 6:
        plt.text(0, 1.0 + dist * aspect, label, horizontalalignment='left',
                 verticalalignment='bottom', transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict)
    elif location == 7:
        plt.text(1.0, 1.0 + dist * aspect, label,
                 horizontalalignment='right', verticalalignment='bottom',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict)
    elif location == 8:
        plt.text(1.0 + dist, 1.0, label,
                 horizontalalignment='left', verticalalignment='top',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict)
    elif location == 9:
        plt.text(1.0 + dist, 0.0, label,
                 horizontalalignment='left', verticalalignment='bottom',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict)
    elif location == 10:
        plt.text(1.0, - dist * aspect, label,
                 horizontalalignment='right', verticalalignment='top',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict)
    elif location == 11:
        plt.text(0.0, -dist * aspect, label, horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes,
                 bbox=boxdict, fontdict=fontdict)
    elif location == 12:
        plt.text(-dist, 0.0, label, horizontalalignment='right',
                 verticalalignment='bottom', transform=ax.transAxes,
                 bbox=boxdict, fontdict=fontdict)
    else:
        raise ValueError("Other corners not defined.")

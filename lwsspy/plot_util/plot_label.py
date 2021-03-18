from typing import Union
import matplotlib
import matplotlib.pyplot as plt
import lwsspy as lpy


def plot_label(ax: matplotlib.axes.Axes, label: str, aspect: float = 1,
               location: int = 1, dist: float = 0.025,
               box: Union[bool, dict] = True, fontdict: dict = {},
               **kwargs):
    """Plots label one of the corners of the plot.

    Plot locations are set as follows::

           6   14   7 
            --------
         5 |1      2|  8
        13 |        | 15
        12 |3      4|  9
            --------
          11   16  10


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
    if type(box) is bool:
        if box:
            boxdict = {'facecolor': 'w', 'edgecolor': 'k'}
        else:
            boxdict = {'facecolor': 'none', 'edgecolor': 'none'}
    else:
        boxdict = box

    # Get aspect of the axes
    aspect = 1.0/lpy.get_aspect(ax)

    # Inside
    if location == 1:
        plt.text(dist, 1.0 - dist * aspect, label, horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict, **kwargs)
    elif location == 2:
        plt.text(1.0 - dist, 1.0 - dist * aspect, label,
                 horizontalalignment='right', verticalalignment='top',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict, **kwargs)
    elif location == 3:
        plt.text(dist, dist * aspect, label, horizontalalignment='left',
                 verticalalignment='bottom', transform=ax.transAxes,
                 bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 4:
        plt.text(1.0 - dist, dist * aspect, label,
                 horizontalalignment='right', verticalalignment='bottom',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict, **kwargs)
    # Outside
    elif location == 5:
        plt.text(dist, 1.0, label, horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict, **kwargs)
    elif location == 6:
        plt.text(0, 1.0 + dist * aspect, label, horizontalalignment='left',
                 verticalalignment='bottom', transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict, **kwargs)
    elif location == 7:
        plt.text(1.0, 1.0 + dist * aspect, label,
                 horizontalalignment='right', verticalalignment='bottom',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict, **kwargs)
    elif location == 8:
        plt.text(1.0 + dist, 1.0, label,
                 horizontalalignment='left', verticalalignment='top',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict, **kwargs)
    elif location == 9:
        plt.text(1.0 + dist, 0.0, label,
                 horizontalalignment='left', verticalalignment='bottom',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict, **kwargs)
    elif location == 10:
        plt.text(1.0, - dist * aspect, label,
                 horizontalalignment='right', verticalalignment='top',
                 transform=ax.transAxes, bbox=boxdict,
                 fontdict=fontdict, **kwargs)
    elif location == 11:
        plt.text(0.0, -dist * aspect, label, horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes,
                 bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 12:
        plt.text(-dist, 0.0, label, horizontalalignment='right',
                 verticalalignment='bottom', transform=ax.transAxes,
                 bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 13:
        plt.text(-dist, 0.5, label, horizontalalignment='right',
                 verticalalignment='center_baseline', transform=ax.transAxes,
                 bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 14:
        plt.text(0.5, 1.0 + dist * aspect, label, horizontalalignment='center',
                 verticalalignment='bottom', transform=ax.transAxes,
                 bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 15:
        plt.text(1 + dist, 0.5, label, horizontalalignment='left',
                 verticalalignment='center_baseline', transform=ax.transAxes,
                 bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 16:
        plt.text(0.5, -dist * aspect, label, horizontalalignment='center',
                 verticalalignment='top', transform=ax.transAxes,
                 bbox=boxdict, fontdict=fontdict, **kwargs)
    else:
        raise ValueError("Other corners not defined.")

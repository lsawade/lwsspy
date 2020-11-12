import matplotlib
import matplotlib.pyplot as plt


def plot_label(ax: matplotlib.axes.Axes, label: str, aspect: float = 1,
               location: int = 1, dist: float = 0.025, box: bool = True,
               fontdict: dict = {}):
    """Plots label one of the corners of the plot.

    .. code::

        1-----2
        |     |
        3-----4


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
    """
    if box:
        boxdict = {'facecolor': 'w', 'edgecolor': 'k'}
    else:
        boxdict = {}

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
    else:
        raise ValueError("Other corners not defined.")

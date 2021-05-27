import matplotlib
import numpy as np


def get_limits(ax: matplotlib.axes.Axes):
    """Gets the limits of all lines in the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        given axes


    """

    # Get all lines in an axes
    lines = ax.lines

    # Get the limits of each one of them
    xmins, xmaxs = [], []
    ymins, ymaxs = [], []
    for _line in lines:
        x = _line.get_xdata()
        y = _line.get_ydata()
        xmins.append(np.min(x))
        xmaxs.append(np.max(x))
        ymins.append(np.min(y))
        ymaxs.append(np.max(y))

    # Get overall mins/maxs
    xmin = np.min(xmins)
    xmax = np.max(xmaxs)
    ymin = np.min(ymins)
    ymax = np.max(ymaxs)

    return xmin, xmax, ymin, ymax

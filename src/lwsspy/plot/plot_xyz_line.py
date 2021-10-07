import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable


def plot_xyz_line(x, y, z, *args, **kwargs):
    """Plot multicolored lines by passing norm and cmap to a LineCollection/

    Mosly taken from matplotlib tutorial on multicolored lines (`MPL`_), just
    streamlined here.

    .. _MPL: https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    Parameters
    ----------
    x : ArrayLike
        x values
    y : ArrayLike
        y values
    z : ArrayLike
        z values

    Raises
    ------
    ValueError
        Errors when x,y,z don' have the same shape

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.04.09 12.50

    """

    # Check if vectors have the same length:
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, and z mustt have he same size")

    # Create Line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create LineCollection
    lines = LineCollection(segments, *args, array=z, **kwargs)

    # Get current axes
    ax = plt.gca()
    ax.add_collection(lines)

    # autoscale since the add_collection does not autoscale
    ax.autoscale()

    # Create a scalarmappable for colorbar purpose
    sm = ScalarMappable(cmap=lines.cmap, norm=lines.norm)

    return lines, sm

import numpy as np
from matplotlib.axes import Axes


def get_aspect(ax: Axes) -> float:
    """Returns the aspect ratio of an axes in a figure. This works around the 
    problem of matplotlib's ``ax.get_aspect`` returning 

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object

    Returns
    -------
    float
        aspect ratio

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.01.20 11.30

    """

    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = np.diff(ax.get_ylim())[0] / np.diff(ax.get_xlim())[0]

    return disp_ratio / data_ratio

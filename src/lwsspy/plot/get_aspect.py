import numpy as np
from matplotlib.axes import Axes


def get_aspect(ax: Axes) -> float:
    """Returns the aspect ratio of an axes in a figure. This works around the 
    problem of matplotlib's ``ax.get_aspect`` returning strings if set to 
    'equal' for example

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

    return disp_ratio

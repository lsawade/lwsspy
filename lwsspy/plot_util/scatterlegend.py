from typing import Callable, DefaultDict, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Normalize, Colormap
from matplotlib.lines import Line2D
from matplotlib.legend import Legend


def scatterlegend(
        values,
        cmap: Optional[Colormap] = None,
        norm: Optional[Normalize] = None,
        sizefunc: Union[Callable, float] = 5,
        handletextpad: float = -2.0,
        fmt: str = '{0:5.2f}',
        lkw=dict(marker='o', markeredgecolor="k", lw=0.2),
        orientation: str = 'h',
        yoffset: float = -50,
        * args, **kwargs) -> Legend:
    """Creates legend of scatter values parsed to function, including a color
    defined by cmap and norm.

    Parameters
    ----------
    values : Iterable
        Values to be put in the legend
    cmap : Optional[Colormap], optional
        Colormap, by default None
    norm : Optional[Normalize], optional
        Norm, by default None
    sizefunc : Union[Callable, float], optional
        Function to define the size of the markers, or float to define size,
        by default 5
    handletextpad: float, optional
        Use to adjust the location of the text underneath the labels. Positive
        values shift the text to the right, default
        -2.0
    fmt : str, optional
        Format specifier, by default '{0:5.2f}'
    lkw : marker dictionary, optional
        dictionary describing the looks of a marker should probably be the same 
        as the one parsed to ``scatter``, 
        by default dict(marker='o', markeredgecolor="k", lw=0.25)
    orientation : str, optional, ['h', 'v']
        `h` for horizonatal, 'v' for vertical, by default 'h'
    yoffset: float
        offset of loegend text, different for png and pdf outputs, default -50


    Returns
    -------
    Legend
        legend
    """

    # Get handles and labels
    handles, labels = [], []

    # For each value
    for v in values:

        # Get markersize from float or functions
        if isinstance(sizefunc, float):
            ms = sizefunc
        else:
            ms = np.sqrt(sizefunc(np.abs(v)))

        # Create handle
        h = Line2D([0], [0], ls="", color=cmap(norm(v)), ms=ms, **lkw)

        # Save handle and label
        handles.append(h)
        labels.append(fmt.format(v))

    # Check how the legend is to be oriented
    if orientation == 'h':
        legend = plt.legend(
            handles, labels, *args, ncol=len(values),
            columnspacing=1.0,
            handletextpad=handletextpad,
            **kwargs
        )

        # Adjust text height
        for txt, line in zip(legend.get_texts(), legend.get_lines()):
            txt.set_ha("center")  # horizontal alignment of text item)
            txt.set_y(yoffset)

    elif orientation == 'v':
        legend = plt.legend(
            handles, labels, *args, ncol=1,
            **kwargs
        )

    return legend

from typing import List, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import colors as cl
from matplotlib.cm import ScalarMappable


def figcolorbar(fig: Figure, axes: List[Axes], cmap='viridis',
                vmin: Union[float, None] = None,
                vmax: Union[float, None] = None,
                **kwargs):
    """Creates generic colorbar for list of subplots in figure.

    Args:
        fig (Figure): Figure
        axes (List[Axes]): List of axes (subplots)
        cmap (str, optional): colormap name. Defaults to 'viridis'.
        vmin (Union[float, None], optional): Smallest value. Defaults to None.
        vmax (Union[float, None], optional): Largest values . Defaults to None.
        **kwargs: parsed to colorbar function

    Returns:
        colorbar handle

    Last modified: Lucas Sawade, 2020.09.15 15.30 (lsawade@princeton.edu)
    """

    norm = cl.Normalize(vmin=vmin, vmax=vmax, clip=False)
    c = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, **kwargs)
    return c

from typing import List
import matplotlib.pyplot as plt
import numpy as np


def pick_colors_from_cmap(N: int, cmap: str = 'viridis') -> List[tuple]:
    """Picks N uniformly distributed colors from a given colormap.

    Parameters
    ----------
    N : int
        Number of wanted colors
    cmap : str, optional
        name of the colormap to pick from, by default 'viridis'


    Returns
    -------
    List[tuple]
        List of color tuples.


    See Also
    --------
    lwsspy.plot_util.update_colorcycler.update_colorcycler : Updates the colors
        used in new lines/scatter points etc.

    """

    # Get cmap
    colormap = plt.get_cmap(cmap)

    # Pick
    colors = colormap(np.linspace(0, 1, N))

    return colors


import matplotlib.pyplot as plt
import lwsspy as lpy


def update_colorcycler(N: int, cmap: str = 'viridis',
                       ax: None or plt.Axes = None):
    """Update color cycler to better automate and choose the colors of certain things

    Parameters
    ----------
    N : int
        Number of colors wanted to cycle through
    cmap : str, optional
        name of the colormap to cycle through, by default 'viridis'
    ax : None or plt.Axes, optional
        Optional Axes to update the color cycler of, by default None

    See Also
    --------
    lwsspy.plot.pick_colors_from_cmap.pick_colors_from_cmap


    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.13 14.30

    """

    # Get Colors
    colors = lpy.pick_colors_from_cmap(N, cmap=cmap)

    # Set axes
    if ax is None:
        ax = plt.gca()

    # Set axes colorcycler
    ax.set_prop_cycle(plt.cycler('color', colors))

import matplotlib.pyplot as plt


def hdaxes(ax=None):
    """Creates base axes for a Hudson skewed-diamond-plot. 
    See Vavryčuk 2015 for a review on Moment tensor decomposition and definition
    of the uv-plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to draw the diamond plot into. If ``None`` new figure and 
        axes will be created, by default ``None``

    Returns
    -------
    matplotlib.axes.Axes
        Returns axes to draw into
    """

    # Create axes
    if ax is None:
        ax = plt.axes()
    # Remove unnecessary things
    plt.axis('equal')
    plt.axis('off')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Plot crosslines
    plt.plot([0, 0], [-1, 1], "k--", lw=0.75)
    plt.plot([-1, 1], [0, 0], "k--", lw=0.75)
    # plt.plot([-4/3, 4/3], [-1/3, 1/3], 'k')

    # Plot outtline
    plt.plot([-4/3, 0], [-1/3, -1], 'k')
    plt.plot([0, 4/3], [-1, 1/3], 'k')
    plt.plot([4/3, 0], [1/3, 1], 'k')
    plt.plot([0, -4/3], [1, -1/3], 'k')

    # Annotations
    plt.text(1.05, 0, "CLVD-", horizontalalignment='left',
             verticalalignment='center_baseline')
    plt.text(0, 1.025, "ISO", horizontalalignment='center',
             verticalalignment='bottom')

    return ax

import matplotlib.pyplot as plt


def ztaxes(ax=None):
    """Creates base axes for a zerotrace assessment plot. Global CMTs are 
    inverted using a Zero Trace condition, meaning there is no isotropic
    component. Hence, we can plot the DC and CLVD components in a 
    single plot. This is an alternative for the uv-plot as defined by 
    Hudson (1989), and presented many times by, eg., Vavryčuk 2015. 
    See the 2015 for a review on Moment tensor decomposition and 
    definition of the uv-plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to draw the ztaxes into. If ``None`` new figure and 
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
    ax.set_aspect('equal', 'box')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel(r"$M_{CLVD}$")
    plt.ylabel(r"$M_{DC}$")

    # Plot crosslines
    plt.plot([0, 0], [-1, 1], "k--", lw=0.75)
    plt.plot([-1, 1], [0, 0], "k--", lw=0.75)

    # Plot outtline
    # plt.plot([-1, 1], [-1, -1], 'k')
    # plt.plot([1, 1], [-1, 1], 'k')
    # plt.plot([1, -1], [1, 1], 'k')
    # plt.plot([-1, -1], [1, -1], 'k')

    # Annotations
    # plt.text(1.05, 0, r"$M_{CLVD}$", horizontalalignment='left',
    #          verticalalignment='center_baseline')
    # plt.text(0, 1.025, r"$M_{DC}$", horizontalalignment='center',
    #          verticalalignment='bottom')

    return ax

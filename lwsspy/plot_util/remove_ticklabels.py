from matplotlib.axes import Axes

def remove_xticklabels(ax: Axes):
    """Removes xticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.axes.xaxis.set_ticklabels([])

def remove_yticklabels(ax: Axes):
    """Removes yticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.axes.yaxis.set_ticklabels([])

from matplotlib.axes import Axes


def remove_xticklabels(ax: Axes):
    """Removes xticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(labelbottom=False)


def remove_yticklabels(ax: Axes):
    """Removes yticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(labelleft=False)

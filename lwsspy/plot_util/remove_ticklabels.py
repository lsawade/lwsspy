from matplotlib.axes import Axes


def remove_xticklabels(ax: Axes):
    """Removes xticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(labelbottom=False, labeltop=False)


def remove_yticklabels(ax: Axes):
    """Removes yticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(labelleft=False, labelright=False)


def remove_ticklabels(ax: Axes):
    """Removes xticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(labelbottom=False, labeltop=False,
                   labelleft=False, labelright=False)


def remove_ticks(ax: Axes):
    """Removes xticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(bottom=False, top=False,
                   left=False, right=False)
